import streamlit as st
import pandas as pd
from PIL import Image
import io, time, json, requests as req
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

st.set_page_config(page_title="TopoS Annotation", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { 
    min-width: 380px !important; 
    max-width: 380px !important;
    transform: none !important;
}
[data-testid="stSidebarContent"] { padding: 1rem; }
section[data-testid="stSidebar"] > div { width: 380px !important; }
</style>
""", unsafe_allow_html=True)
SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID = st.secrets["TOPOS_FOLDER_ID"]
PATCHES_FOLDER_ID = st.secrets["PATCHES_FOLDER_ID"]
FEATURES = ['clearing', 'water', 'wood_church', 'bridge']
ANNOTATOR_ID = '000'  # hardcoded for test app

FEATURE_INFO = {
    'clearing': {
        'title': 'Forest Clearings',
        'folder': 'clearing',
        'description': (
            'Forest clearings are a key component of the agricultural landscape. '
            'They are characterized, in most cases, by thin vertical lines perpendicular '
            'to shorter horizontal lines set against a blank background. '
            'Cleared areas often lack boundary lines and their shapes are irregular.'
        ),
    },
    'water': {
        'title': 'Water Bodies',
        'folder': 'water',
        'description': (
            'For the purposes of this activity, we want to identify ponds, lakes, and similar '
            'bodies of water (rather than streams or rivers). '
            'Water bodies are bounded and are depicted by a series of nested polygons.'
        ),
    },
    'wood_church': {
        'title': 'Wooden Churches',
        'folder': 'wood church',
        'description': (
            'The map depicts a range of religious sites. We want to identify wooden churches. '
            'They are depicted with empty circles topped by a vertical cross.'
        ),
    },
    'bridge': {
        'title': 'Bridges',
        'folder': 'bridge',
        'description': (
            'Bridges over water, land, railways, and roads are all depicted by a pair of '
            'parallel lines whose ends curl outward. They vary in size and orientation.'
        ),
    },
}

@st.cache_resource
def get_creds():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"]), scopes=SCOPES)
    return creds

@st.cache_resource
def get_drive_service():
    return build('drive', 'v3', credentials=get_creds())

def get_token():
    creds = get_creds()
    if not creds.valid:
        creds.refresh(Request())
    return creds.token

def find_file(svc, name, fid):
    r = svc.files().list(
        q=f"name='{name}' and '{fid}' in parents and trashed=false",
        fields="files(id)").execute()
    f = r.get('files', [])
    return f[0]['id'] if f else None

def dl_csv(svc, file_id):
    req2 = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req2)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf, dtype=str, engine='python')

def ul_csv(df, file_id, name, folder_id):
    from googleapiclient.http import MediaIoBaseUpload as MU
    svc = get_drive_service()
    for attempt in range(3):
        try:
            buf = io.BytesIO(df.to_csv(index=False).encode('utf-8'))
            media = MU(buf, mimetype='text/csv', resumable=False)
            if file_id:
                svc.files().update(fileId=file_id, media_body=media).execute()
                return file_id
            else:
                f = svc.files().create(
                    body={'name': name, 'parents': [folder_id]},
                    media_body=media, fields='id'
                ).execute()
                return f['id']
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                st.warning(f"⚠️ Auto-save failed: {e}")
                return file_id

def dl_img(svc, file_id):
    for attempt in range(3):
        try:
            req2 = svc.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, req2)
            done = False
            while not done:
                _, done = dl.next_chunk()
            buf.seek(0)
            img = Image.open(buf)
            img.load()
            return img
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                raise e

@st.cache_data(ttl=300)
def load_patch_index(_svc, pfid):
    files, pt = [], None
    while True:
        kw = dict(q=f"'{pfid}' in parents and trashed=false",
                  fields="nextPageToken,files(id,name)", pageSize=1000)
        if pt: kw['pageToken'] = pt
        r = _svc.files().list(**kw).execute()
        files.extend(r.get('files', []))
        pt = r.get('nextPageToken')
        if not pt: break
    return {f['name']: f['id'] for f in files}

def is_true(v): return str(v).strip() in ('True','true','1','TRUE')
def count_true(s): return int(s.apply(is_true).sum())

def init_state():
    for k, v in {
        'initialized':False,'service':None,
        'assignments_df':None,'my_patches':None,'csv_file_id':None,
        'csv_filename':None,'feature_idx':0,'review_mode':None,
        'review_idx':0,'review_patches':None,'review_total':0,'patch_start':None,
        'flagging':False,'patch_index':None,'saving':False,'show_congrats':None,'all_done':False,
        'annotations_folder_id':None,'patches_folder_id':None,
    }.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()

def get_counts(mp):
    labeled = mp['label'].isin(['present','absent'])
    return {
        'total':len(mp),
        'done':int(labeled.sum()),
        'present':int((mp['label']=='present').sum()),
        'absent':int((mp['label']=='absent').sum()),
        'skipped':int(mp['was_skipped'].apply(is_true).sum()),
        'flagged':int(mp['was_flagged'].apply(is_true).sum()),
    }

def get_unlabeled(mp): return mp[mp['label'].isna() | (mp['label'] == '')].reset_index(drop=True)

def load_feature(svc, aid, fidx, adf, ann_fid):
    feature = FEATURES[fidx]
    fname = f'annotator_{aid}_{feature}.csv'
    mp = adf[(adf['annotator_id']==aid)&(adf['feature']==feature)].copy().reset_index(drop=True)
    for col in ['label','original_label','final_label','time_seconds','review_time_seconds']:
        if col not in mp.columns: mp[col] = None
        mp[col] = mp[col].astype('object')
    mp['was_skipped'] = 'False'
    mp['was_flagged'] = 'False'
    file_id = find_file(svc, fname, ann_fid)
    if file_id:
        saved = dl_csv(svc, file_id)
        cols = ['patch_id']+[c for c in [
            'label','original_label','final_label','time_seconds',
            'review_time_seconds','was_skipped','was_flagged'] if c in saved.columns]
        mg = mp.merge(saved[cols], on='patch_id', how='left', suffixes=('','_saved'))
        for col in ['label','original_label','final_label','time_seconds',
                    'review_time_seconds','was_skipped','was_flagged']:
            sc = col+'_saved'
            if sc in mg.columns:
                mask = mg[sc].notna() & (mg[sc]!='')
                mg.loc[mask, col] = mg.loc[mask, sc]
                mg = mg.drop(columns=[sc])
        mp = mg
    return mp, file_id, fname

def save_all(mp, fid, fname, ann_fid):
    new_id = ul_csv(mp, fid, fname, ann_fid)
    if new_id and new_id != fid:
        st.session_state['csv_file_id'] = new_id

def upd(mp, pid, label, elapsed, is_review=False, orig=None):
    idx = mp[mp['patch_id']==pid].index[0]
    mp.at[idx,'label'] = label
    # was_skipped and was_flagged are permanent history — never cleared
    if elapsed: mp.at[idx,'review_time_seconds' if is_review else 'time_seconds'] = str(elapsed)
    if is_review:
        mp.at[idx,'final_label'] = label
    if orig:
        mp.at[idx,'original_label'] = orig
        mp.at[idx,'was_flagged'] = 'True'
    else:
        if not is_review:
            mp.at[idx,'original_label'] = label
            mp.at[idx,'final_label'] = label
    return mp

def skip_p(mp, pid, elapsed):
    idx = mp[mp['patch_id']==pid].index[0]
    mp.at[idx,'was_skipped']='True'
    mp.at[idx,'label']='skipped'  # mark as skipped so it's excluded from unlabeled
    if elapsed: mp.at[idx,'time_seconds']=str(elapsed)
    return mp

# ── AUTO INIT ──
ss = st.session_state
if not ss['initialized']:
    with st.spinner("Loading..."):
        try:
            svc = get_drive_service()
            asid = find_file(svc, 'patch_assignments.csv', FOLDER_ID)
            adf  = dl_csv(svc, asid)
            adf['annotator_id'] = adf['annotator_id'].astype(str).str.zfill(3)

            r = svc.files().list(
                q=f"name='annotations' and '{FOLDER_ID}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                fields="files(id)").execute()
            af = r.get('files',[])
            if af:
                ann_fid = af[0]['id']
            else:
                ann_fid = svc.files().create(
                    body={'name':'annotations','mimeType':'application/vnd.google-apps.folder','parents':[FOLDER_ID]},
                    fields='id').execute()['id']

            pr = svc.files().list(
                q=f"name='patches' and '{FOLDER_ID}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                fields="files(id)").execute()
            pf = pr.get('files',[])
            pfid = pf[0]['id'] if pf else PATCHES_FOLDER_ID
            pi = load_patch_index(svc, pfid)
            mp, fid, fname = load_feature(svc, ANNOTATOR_ID, 0, adf, ann_fid)

            ss.update({
                'initialized':True,'service':svc,
                'assignments_df':adf,'my_patches':mp,'csv_file_id':fid,'csv_filename':fname,
                'feature_idx':0,'review_mode':None,'review_idx':0,'review_patches':None,
                'patch_start':time.time(),'patch_index':pi,
                'annotations_folder_id':ann_fid,'patches_folder_id':pfid,
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error loading: {e}")
            st.stop()

# ── ANNOTATION ──
svc=ss['service']; feature=FEATURES[ss['feature_idx']]

# All done screen
if ss.get('all_done'):
    st.balloons()
    st.success(f"""
## 🎉 Congratulations, Annotator {ANNOTATOR_ID}!

You have completed **all {len(FEATURES)} feature rounds** for your assigned patches.

Thank you for contributing to Project Imperiia — TopoS. Your annotations help digitize the Military-Topographic Survey of European Russia (MTSER) and bring 19th-century maps to life for modern research.

Your results have been saved automatically. You may now close this window.
    """)
    st.stop()

# Page header
st.markdown("#### 🗺️ Welcome to Project Imperiia — TopoS &nbsp; <small style='font-weight:normal;'>— You are helping digitize the Military-Topographic Survey of European Russia (MTSER), a 19th-century map series.</small>", unsafe_allow_html=True)

# Always get fresh data from session state
mp=ss['my_patches']; c=get_counts(ss['my_patches'])
in_review = ss['review_mode'] is not None
progress = f"{c['done']}/{c['total']}"

st.markdown(
    f"**👤 Annotator {ANNOTATOR_ID}** | **📋 {feature}** | **📊 {progress}**"
    f"&nbsp;&nbsp; ✅**{c['present']}** ❌**{c['absent']}** ⏭**{c['skipped']}** 🚩**{c['flagged']}**"
)

if ss.get('show_congrats'):
    completed = ss['show_congrats']
    st.success(f"🎉 Congratulations! You completed **Feature {completed} of {len(FEATURES)}**. Moving to Feature {completed + 1}...")
    ss['show_congrats'] = None

def elapsed(): return round(time.time()-ss['patch_start'],1) if ss['patch_start'] else None
def save():
    ss['saving'] = True
    st.rerun()

# Execute pending save
if ss.get('saving', False):
    with st.spinner("💾 Saving... please wait"):
        save_all(ss['my_patches'],ss['csv_file_id'],ss['csv_filename'],ss['annotations_folder_id'])
    ss['saving'] = False
    ss['patch_start'] = time.time()
    st.rerun()

def adv_feat():
    ni = ss['feature_idx'] + 1
    completed = ss['feature_idx'] + 1
    mp2, fid, fn = load_feature(svc, ANNOTATOR_ID, ni, ss['assignments_df'], ss['annotations_folder_id'])
    ss.update({'feature_idx': ni, 'review_mode': None, 'my_patches': mp2,
               'csv_file_id': fid, 'csv_filename': fn, 'patch_start': time.time(),
               'show_congrats': completed})
    st.rerun()

def enter_rev(rt):
    fmp = ss['my_patches']
    if rt=='skipped':
        ps = fmp[fmp['label']=='skipped']
    else:
        def needs_flagged_review(row):
            fl = str(row.get('final_label', ''))
            return is_true(row.get('was_flagged','')) and fl in ('', 'nan', 'None', '<NA>')
        ps = fmp[fmp.apply(needs_flagged_review, axis=1)]
    if len(ps)>0:
        ss['review_mode']=rt; ss['review_idx']=0
        ss['review_patches']=ps.reset_index(drop=True)
        ss['review_total']=len(ps)
        ss['patch_start']=time.time()
        st.rerun()
    else: nxt_rev(rt)

def nxt_rev(cur):
    if cur=='skipped': enter_rev('flagged')
    elif ss['feature_idx']<len(FEATURES)-1: adv_feat()
    else:
        ss['all_done'] = True
        st.rerun()

INSTRUCTIONS = """
**Welcome to Project Imperiia — TopoS**

You are helping digitize the Military-Topographic Survey of European Russia (MTSER), a 19th-century map series.

---

**Your task:** For each map patch, decide whether the target feature is present.

---

**Buttons:**
- ✅ **Present** — The feature is clearly visible in this patch
- ❌ **Absent** — The feature is not visible
- ⏭ **Skip** — You are unsure. You will review skipped patches at the end of each feature round
- 🚩 **Flag** — Make your best guess (Present or Absent) but mark it for review. Use this when you are uncertain but want to keep going

---

**Tips:**
- A feature counts as Present even if only partially visible
- When in doubt between Skip and Flag, use Flag — it records your best guess
- Your progress saves automatically after every click
"""

@st.cache_data(ttl=3600)
SCREENCAPTURES_FOLDER = 'feature screencaptures'

@st.cache_data(ttl=3600)
def get_screencaptures_folder_id(_svc, parent_folder_id):
    r = _svc.files().list(
        q=f"name='feature screencaptures' and '{parent_folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
        fields="files(id)"
    ).execute()
    f = r.get('files', [])
    return f[0]['id'] if f else None

@st.cache_data(ttl=3600)
def get_subfolder_id(_svc, parent_id, name):
    r = _svc.files().list(
        q=f"name='{name}' and '{parent_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
        fields="files(id)"
    ).execute()
    f = r.get('files', [])
    return f[0]['id'] if f else None

@st.cache_data(ttl=3600)
def load_sample_img(_svc, feature, folder_id):
    try:
        sc_fid = get_screencaptures_folder_id(_svc, folder_id)
        if not sc_fid:
            return None
        info = FEATURE_INFO.get(feature, {})
        subfolder_name = info.get('folder', feature)
        sub_fid = get_subfolder_id(_svc, sc_fid, subfolder_name)
        if not sub_fid:
            return None
        # Get first image in subfolder
        r = _svc.files().list(
            q=f"'{sub_fid}' in parents and trashed=false and mimeType contains 'image/'",
            fields="files(id, name)",
            pageSize=1,
            orderBy="name"
        ).execute()
        imgs = r.get('files', [])
        if not imgs:
            return None
        return dl_img(_svc, imgs[0]['id'])
    except Exception:
        return None

def show_img(pid):
    with st.sidebar:
        st.markdown("## 📖 Annotation Guide")
        info = FEATURE_INFO.get(feature, {})
        st.markdown(f"Your task is to identify **{info.get('title', feature)}** in each map patch. Below is a sample reference image to guide your judgment.")
        if info.get('description'):
            st.info(info['description'])
        st.divider()
        sample = load_sample_img(svc, feature, FOLDER_ID)
        if sample:
            st.image(sample, use_container_width=True)
        else:
            st.info("Sample image coming soon.")
        st.divider()
        st.markdown("""
**For each patch:**

- ✅ **Present** — The feature is visible in this patch
- ❌ **Absent** — The feature is not visible
- ⏭ **Skip** — Unsure. You will review skipped patches at the end of each feature
- 🚩 **Flag** — Make your best guess but mark for review

**Tips:**
- A feature counts as Present even if only partially visible
- When in doubt between Skip and Flag, use Flag
- Progress saves automatically after every click
        """)

    # Main: patch image
    if pid not in ss['patch_index']:
        st.warning("⚠️ Image not uploaded yet — you can still label below.")
        return
    try:
        img = dl_img(svc, ss['patch_index'][pid])
        st.image(img, width=480)
    except Exception:
        st.warning("⚠️ Image not available — you can still label below.")

if ss['review_mode'] is not None:
    rt=ss['review_mode']; ps=ss['review_patches']; idx=ss['review_idx']
    if idx>=len(ps): nxt_rev(rt); st.stop()
    row=ps.iloc[idx]; pid=row['patch_id']

    # Compute header counts from latest data
    fresh_mp = ss['my_patches']
    def has_final_label(val):
        return str(val) not in ('', 'nan', 'None', '<NA>')
    if rt=='skipped':
        n_total   = int(fresh_mp['was_skipped'].apply(is_true).sum())
        n_labeled = int((fresh_mp['was_skipped'].apply(is_true) & fresh_mp['label'].isin(['present','absent'])).sum())
        header = f"⏭ Review {n_labeled}/{n_total} skipped patches"
    else:
        n_total   = int(fresh_mp['was_flagged'].apply(is_true).sum())
        n_labeled = int((fresh_mp['was_flagged'].apply(is_true) & fresh_mp['final_label'].apply(has_final_label)).sum())
        header = f"🚩 Review {n_labeled}/{n_total} flagged patches"
    st.markdown(f"#### {header}")
    if rt=='flagged':
        orig=str(row.get('original_label',''))
        if orig not in ('nan','<NA>','','None'): st.info(f"🏷️ Original: **{orig}**")
    show_img(pid)
    c1,c2=st.columns(2)
    with c1:
        if st.button("✅ Present", type="primary", use_container_width=True, disabled=ss["saving"]):
            ss['my_patches']=upd(ss['my_patches'],pid,'present',elapsed(),is_review=True)
            ss['review_idx']+=1
            save()
    with c2:
        if st.button("❌ Absent", type="secondary", use_container_width=True, disabled=ss["saving"]):
            ss['my_patches']=upd(ss['my_patches'],pid,'absent',elapsed(),is_review=True)
            ss['review_idx']+=1
            save()
else:
    ul=get_unlabeled(ss['my_patches'])
    if len(ul)==0: enter_rev('skipped'); st.stop()
    pid=ul.iloc[0]['patch_id']; show_img(pid)
    if ss.get('saving', False):
        st.info("💾 Saving... please wait.")
    elif ss['flagging']:
        st.warning("🚩 Flag as which? Choose your best guess — marked for review.")
        c1,c2,c3=st.columns(3)
        with c1:
            if st.button("🚩 Flag as Present", use_container_width=True, disabled=ss["saving"]):
                ss['my_patches']=upd(ss['my_patches'],pid,'present',elapsed(),orig='present')
                ss['flagging']=False; save()
        with c2:
            if st.button("🚩 Flag as Absent", use_container_width=True, disabled=ss["saving"]):
                ss['my_patches']=upd(ss['my_patches'],pid,'absent',elapsed(),orig='absent')
                ss['flagging']=False; save()
        with c3:
            if st.button("Cancel", use_container_width=True):
                ss['flagging']=False; st.rerun()
    else:
        c1,c2,c3,c4=st.columns(4)
        with c1:
            if st.button("✅ Present", type="primary", use_container_width=True, disabled=ss["saving"]):
                ss['my_patches']=upd(ss['my_patches'],pid,'present',elapsed())
                save()
        with c2:
            if st.button("❌ Absent", type="secondary", use_container_width=True, disabled=ss["saving"]):
                ss['my_patches']=upd(ss['my_patches'],pid,'absent',elapsed())
                save()
        with c3:
            if st.button("⏭ Skip", use_container_width=True, disabled=ss["saving"]):
                ss['my_patches']=skip_p(ss['my_patches'],pid,elapsed())
                save()
        with c4:
            if st.button("🚩 Flag", use_container_width=True, disabled=ss["saving"]):
                ss['flagging']=True; st.rerun()
