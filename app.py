import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import time
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# ── Page config ──
st.set_page_config(
    page_title="TopoS Annotation",
    page_icon="🗺️",
    layout="centered"
)

# ── Google Drive setup ──
SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID     = st.secrets["TOPOS_FOLDER_ID"]
PATCHES_FOLDER_ID = st.secrets["PATCHES_FOLDER_ID"]

@st.cache_resource
def get_drive_service():
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def find_file(service, name, folder_id):
    results = service.files().list(
        q=f"name='{name}' and '{folder_id}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None

def download_csv(service, file_id):
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf, dtype=str)

def upload_csv(service, df, file_id, filename, folder_id):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype='text/csv', resumable=True)
    if file_id:
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        file_metadata = {'name': filename, 'parents': [folder_id]}
        service.files().create(body=file_metadata, media_body=media).execute()

def download_image(service, file_id):
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return Image.open(buf)

@st.cache_data(ttl=300)
def load_patch_index(_service, patches_folder_id):
    results = _service.files().list(
        q=f"'{patches_folder_id}' in parents and trashed=false",
        fields="files(id, name)",
        pageSize=1000
    ).execute()
    files = results.get('files', [])
    return {f['name']: f['id'] for f in files}

FEATURES = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']

# ── Session state init ──
def init_state():
    defaults = {
        'logged_in':      False,
        'annotator_id':   None,
        'service':        None,
        'credentials_df': None,
        'assignments_df': None,
        'my_patches':     None,
        'csv_file_id':    None,
        'feature_idx':    0,
        'review_mode':    None,
        'review_idx':     0,
        'review_patches': None,
        'patch_start':    None,
        'flagging':       False,
        'patch_index':    None,
        'annotations_folder_id': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helper: get counts ──
def get_counts(mp):
    return {
        'total':   len(mp),
        'done':    int(mp['label'].notna().sum()),
        'present': int((mp['label'] == 'present').sum()),
        'absent':  int((mp['label'] == 'absent').sum()),
        'skipped': int(mp['was_skipped'].astype(str).isin(['True','true','1']).sum()),
        'flagged': int(mp['was_flagged'].astype(str).isin(['True','true','1']).sum()),
    }

def get_unlabeled(mp):
    return mp[mp['label'].isna()].reset_index(drop=True)

def load_feature(service, annotator_id, feature_idx, assignments_df,
                 annotations_folder_id):
    feature  = FEATURES[feature_idx]
    filename = f'annotator_{annotator_id}_{feature}.csv'

    my_patches = assignments_df[
        (assignments_df['annotator_id'] == annotator_id) &
        (assignments_df['feature'] == feature)
    ].copy()

    for col in ['label', 'original_label', 'final_label',
                'time_seconds', 'review_time_seconds']:
        my_patches[col] = pd.NA
        my_patches[col] = my_patches[col].astype('object')
    for col in ['was_skipped', 'was_flagged']:
        my_patches[col] = False

    file_id = find_file(service, filename, annotations_folder_id)
    if file_id:
        saved = download_csv(service, file_id)
        saved_cols = ['patch_id'] + [c for c in [
            'label', 'original_label', 'final_label',
            'time_seconds', 'review_time_seconds',
            'was_skipped', 'was_flagged'
        ] if c in saved.columns]
        my_patches = my_patches.merge(saved[saved_cols], on='patch_id',
            how='left', suffixes=('', '_saved'))
        for col in ['label', 'original_label', 'final_label',
                    'time_seconds', 'review_time_seconds']:
            sc = col + '_saved'
            if sc in my_patches.columns:
                mask = my_patches[sc].notna()
                my_patches.loc[mask, col] = my_patches.loc[mask, sc]
                my_patches = my_patches.drop(columns=[sc])
        for col in ['was_skipped', 'was_flagged']:
            sc = col + '_saved'
            if sc in my_patches.columns:
                mask = my_patches[sc].notna()
                my_patches.loc[mask, col] = my_patches.loc[mask, sc]
                my_patches = my_patches.drop(columns=[sc])

    return my_patches, file_id, filename

def save_patches(service, my_patches, file_id, filename, annotations_folder_id):
    upload_csv(service, my_patches, file_id, filename, annotations_folder_id)

def update_patch(my_patches, patch_id, label, elapsed,
                 is_review=False, original_label=None):
    idx = my_patches[my_patches['patch_id'] == patch_id].index[0]
    my_patches.at[idx, 'label']       = label
    my_patches.at[idx, 'was_skipped'] = False
    if elapsed is not None:
        col = 'review_time_seconds' if is_review else 'time_seconds'
        my_patches.at[idx, col] = elapsed
    if is_review:
        my_patches.at[idx, 'final_label'] = label
    if original_label is not None:
        my_patches.at[idx, 'original_label'] = original_label
        my_patches.at[idx, 'was_flagged']    = True
    else:
        my_patches.at[idx, 'original_label'] = label
        my_patches.at[idx, 'final_label']    = label
    return my_patches

def skip_patch(my_patches, patch_id, elapsed):
    idx = my_patches[my_patches['patch_id'] == patch_id].index[0]
    my_patches.at[idx, 'was_skipped'] = True
    my_patches.at[idx, 'was_flagged'] = False
    if elapsed is not None:
        my_patches.at[idx, 'time_seconds'] = elapsed
    return my_patches

# ────────────────────────────────────────────────
# LOGIN PAGE
# ────────────────────────────────────────────────
if not st.session_state['logged_in']:
    st.title("🗺️ TopoS Annotation System")
    st.divider()
    st.subheader("🔐 Please log in")

    annotator_id = st.text_input("Annotator ID", placeholder="e.g. 001", max_chars=3)
    password     = st.text_input("Password", type="password", placeholder="5-digit password")
    login_btn    = st.button("Login →", type="primary")

    if login_btn:
        with st.spinner("Connecting to Google Drive..."):
            try:
                service = get_drive_service()

                # Find credentials file
                cred_id = find_file(service, 'annotator_credentials.csv', FOLDER_ID)
                if not cred_id:
                    st.error("Cannot find annotator_credentials.csv in Google Drive.")
                    st.stop()
                credentials_df = download_csv(service, cred_id)

                # Verify login
                aid = annotator_id.strip().zfill(3)
                match = credentials_df[
                    (credentials_df['annotator_id'] == aid) &
                    (credentials_df['password'] == password.strip())
                ]
                if match.empty:
                    st.error("❌ Invalid ID or password. Please try again.")
                    st.stop()

                # Load assignments
                assign_id = find_file(service, 'patch_assignments.csv', FOLDER_ID)
                assignments_df = download_csv(service, assign_id)
                assignments_df['annotator_id'] = assignments_df['annotator_id'].astype(str).str.zfill(3)

                # Find annotations folder
                results = service.files().list(
                    q=f"name='annotations' and '{FOLDER_ID}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                    fields="files(id)"
                ).execute()
                ann_files = results.get('files', [])
                if ann_files:
                    annotations_folder_id = ann_files[0]['id']
                else:
                    # Create annotations folder
                    folder_meta = {'name': 'annotations', 'mimeType': 'application/vnd.google-apps.folder', 'parents': [FOLDER_ID]}
                    folder = service.files().create(body=folder_meta, fields='id').execute()
                    annotations_folder_id = folder['id']

                # Find patches folder
                p_results = service.files().list(
                    q=f"name='patches' and '{FOLDER_ID}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                    fields="files(id)"
                ).execute()
                p_files = p_results.get('files', [])
                patches_folder_id = p_files[0]['id'] if p_files else PATCHES_FOLDER_ID

                # Load patch index
                patch_index = load_patch_index(service, patches_folder_id)

                # Load first feature
                my_patches, csv_file_id, csv_filename = load_feature(
                    service, aid, 0, assignments_df, annotations_folder_id)

                # Store in session
                st.session_state.update({
                    'logged_in':             True,
                    'annotator_id':          aid,
                    'service':               service,
                    'assignments_df':        assignments_df,
                    'my_patches':            my_patches,
                    'csv_file_id':           csv_file_id,
                    'csv_filename':          csv_filename,
                    'feature_idx':           0,
                    'review_mode':           None,
                    'review_idx':            0,
                    'review_patches':        None,
                    'patch_start':           time.time(),
                    'patch_index':           patch_index,
                    'annotations_folder_id': annotations_folder_id,
                    'patches_folder_id':     patches_folder_id,
                })
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

# ────────────────────────────────────────────────
# ANNOTATION PAGE
# ────────────────────────────────────────────────
else:
    ss      = st.session_state
    service = ss['service']
    aid     = ss['annotator_id']
    feature = FEATURES[ss['feature_idx']]
    mp      = ss['my_patches']
    counts  = get_counts(mp)

    # ── Status bar ──
    st.markdown(
        f"**👤 Annotator {aid}** &nbsp;|&nbsp; **📋 {feature}** &nbsp;|&nbsp; "
        f"**📊 {counts['done']+1} / {counts['total']}** &nbsp;&nbsp; "
        f"✅ **{counts['present']}** &nbsp; ❌ **{counts['absent']}** &nbsp; "
        f"⏭ **{counts['skipped']}** &nbsp; 🚩 **{counts['flagged']}**"
    )
    st.divider()

    def get_elapsed():
        if ss['patch_start'] is not None:
            return round(time.time() - ss['patch_start'], 1)
        return None

    def do_save():
        save_patches(service, ss['my_patches'],
                     ss['csv_file_id'], ss['csv_filename'],
                     ss['annotations_folder_id'])

    def advance_feature():
        next_idx = ss['feature_idx'] + 1
        my_patches, csv_file_id, csv_filename = load_feature(
            service, aid, next_idx,
            ss['assignments_df'], ss['annotations_folder_id'])
        ss.update({
            'feature_idx':  next_idx,
            'review_mode':  None,
            'my_patches':   my_patches,
            'csv_file_id':  csv_file_id,
            'csv_filename': csv_filename,
            'patch_start':  time.time(),
        })
        st.rerun()

    def enter_review(review_type):
        if review_type == 'skipped':
            patches = ss['my_patches'][ss['my_patches']['label'].isna()]
        else:
            patches = ss['my_patches'][ss['my_patches']['was_flagged'].astype(str).isin(['True','true','1'])]
        if len(patches) > 0:
            ss['review_mode']    = review_type
            ss['review_idx']     = 0
            ss['review_patches'] = patches.reset_index(drop=True)
            ss['patch_start']    = time.time()
            st.rerun()
        else:
            next_review(review_type)

    def next_review(current_type):
        if current_type == 'skipped':
            enter_review('flagged')
        elif ss['feature_idx'] < len(FEATURES) - 1:
            advance_feature()
        else:
            ss['logged_in'] = False
            st.rerun()

    # ── Show image ──
    def show_image(patch_id):
        if patch_id not in ss['patch_index']:
            st.warning(f"Patch not found: {patch_id}")
            return
        file_id = ss['patch_index'][patch_id]
        try:
            img = download_image(service, file_id)
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    # ── REVIEW MODE ──
    if ss['review_mode'] is not None:
        review_type = ss['review_mode']
        patches     = ss['review_patches']
        idx         = ss['review_idx']

        if idx >= len(patches):
            next_review(review_type)
            st.stop()

        row      = patches.iloc[idx]
        patch_id = row['patch_id']
        color    = '#fff3cd' if review_type == 'skipped' else '#f8d7da'
        emoji    = '⏭' if review_type == 'skipped' else '🚩'

        st.markdown(f"#### {emoji} Reviewing {review_type} — {idx+1} / {len(patches)}")

        if review_type == 'flagged':
            orig = str(row.get('original_label', ''))
            if orig not in ('nan', '<NA>', '', 'None'):
                st.info(f"🏷️ Original label: **{orig}**")

        show_image(patch_id)
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Present", type="primary", use_container_width=True):
                elapsed = get_elapsed()
                ss['my_patches'] = update_patch(
                    ss['my_patches'], patch_id, 'present', elapsed, is_review=True)
                do_save()
                ss['review_idx'] += 1
                ss['patch_start'] = time.time()
                st.rerun()
        with col2:
            if st.button("❌ Absent", type="secondary", use_container_width=True):
                elapsed = get_elapsed()
                ss['my_patches'] = update_patch(
                    ss['my_patches'], patch_id, 'absent', elapsed, is_review=True)
                do_save()
                ss['review_idx'] += 1
                ss['patch_start'] = time.time()
                st.rerun()

    # ── NORMAL MODE ──
    else:
        unlabeled = get_unlabeled(mp)

        if len(unlabeled) == 0:
            enter_review('skipped')
            st.stop()

        patch_id = unlabeled.iloc[0]['patch_id']
        show_image(patch_id)
        st.divider()

        # Flag mode
        if ss['flagging']:
            st.warning("🚩 **Flag as which?** Choose your best guess — marked for review.")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🚩 Flag as Present", use_container_width=True):
                    elapsed = get_elapsed()
                    ss['my_patches'] = update_patch(
                        ss['my_patches'], patch_id, 'present', elapsed,
                        original_label='present')
                    ss['flagging'] = False
                    do_save()
                    ss['patch_start'] = time.time()
                    st.rerun()
            with col2:
                if st.button("🚩 Flag as Absent", use_container_width=True):
                    elapsed = get_elapsed()
                    ss['my_patches'] = update_patch(
                        ss['my_patches'], patch_id, 'absent', elapsed,
                        original_label='absent')
                    ss['flagging'] = False
                    do_save()
                    ss['patch_start'] = time.time()
                    st.rerun()
            with col3:
                if st.button("Cancel", use_container_width=True):
                    ss['flagging'] = False
                    st.rerun()

        # Normal buttons
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("✅ Present", type="primary", use_container_width=True):
                    elapsed = get_elapsed()
                    ss['my_patches'] = update_patch(
                        ss['my_patches'], patch_id, 'present', elapsed)
                    do_save()
                    ss['patch_start'] = time.time()
                    st.rerun()
            with col2:
                if st.button("❌ Absent", type="secondary", use_container_width=True):
                    elapsed = get_elapsed()
                    ss['my_patches'] = update_patch(
                        ss['my_patches'], patch_id, 'absent', elapsed)
                    do_save()
                    ss['patch_start'] = time.time()
                    st.rerun()
            with col3:
                if st.button("⏭ Skip", use_container_width=True):
                    elapsed = get_elapsed()
                    ss['my_patches'] = skip_patch(ss['my_patches'], patch_id, elapsed)
                    do_save()
                    ss['patch_start'] = time.time()
                    st.rerun()
            with col4:
                if st.button("🚩 Flag", use_container_width=True):
                    ss['flagging'] = True
                    st.rerun()
