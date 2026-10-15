"""Streamlit front end for the DAM gateway.

The JWT lives only in server-side session state (never in the URL); reloading the page
signs the user out. Media is streamed by the browser straight from object storage via
short-lived presigned URLs.
"""

import os
import time

import requests
import streamlit as st

API = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "mp4", "mov", "mkv", "avi", "mp3", "wav"]
TYPE_ICONS = {"document": "📄", "image": "🖼️", "video": "🎬", "audio": "🎧"}

st.set_page_config(page_title="Multimodal DAM", page_icon="🗂️", layout="wide")


# ====================================================================== API helpers
class ApiError(Exception):
    pass


def api(method: str, path: str, timeout: float = 30, **kwargs):
    headers = kwargs.pop("headers", {})
    if token := st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        res = requests.request(method, f"{API}{path}", headers=headers, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Gateway unreachable: {exc}") from exc
    if res.status_code == 401 and st.session_state.get("token"):
        st.session_state.clear()
        st.rerun()
    if res.status_code >= 400:
        try:
            detail = res.json().get("detail", res.text)
        except ValueError:
            detail = res.text
        raise ApiError(f"{res.status_code}: {detail}")
    return res.json() if res.content else None


@st.cache_data(ttl=600, show_spinner=False)
def media_url(asset_id: str, _token: str) -> str | None:
    try:
        return api("GET", f"/api/assets/{asset_id}/url")["url"]
    except ApiError:
        return None


def fmt_ts(seconds) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def fmt_size(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def location(hit: dict) -> str:
    if hit.get("page") is not None:
        return f"page {hit['page']}"
    if hit.get("kind") == "caption":
        return "image description"
    if hit.get("start") is not None:
        return f"{fmt_ts(hit['start'])}"
    return ""


def open_player(hit: dict) -> None:
    st.session_state["player"] = {
        "asset_id": hit["asset_id"],
        "filename": hit["filename"],
        "media_type": hit["media_type"],
        "start": hit.get("start") or 0,
        "page": hit.get("page"),
    }


def render_player(slot_key: str) -> None:
    """Shows the selected search hit or citation at its page/timestamp."""
    player = st.session_state.get("player")
    if not player:
        return
    url = media_url(player["asset_id"], st.session_state["token"])
    with st.container(border=True):
        head, close = st.columns([6, 1])
        head.markdown(f"**{TYPE_ICONS.get(player['media_type'], '')} {player['filename']}** · {location(player)}")
        if close.button("Close", key=f"close-{slot_key}"):
            st.session_state.pop("player")
            st.rerun()
        if not url:
            st.warning("Could not get a media link.")
        elif player["media_type"] == "video":
            st.video(url, start_time=int(player["start"]))
        elif player["media_type"] == "audio":
            st.audio(url, start_time=int(player["start"]))
        elif player["media_type"] == "image":
            st.image(url, use_column_width=True)
        else:
            page = player.get("page") or 1
            st.link_button(f"Open PDF at page {page}", f"{url}#page={page}")


def jump_button(hit: dict, key: str) -> None:
    label = {"video": "▶ Play", "audio": "▶ Play", "image": "🖼 View", "document": "📄 Open"}[hit["media_type"]]
    if hit.get("start") is not None and hit["media_type"] in ("video", "audio"):
        label += f" at {fmt_ts(hit['start'])}"
    elif hit.get("page") is not None:
        label += f" page {hit['page']}"
    st.button(label, key=key, on_click=open_player, args=(hit,))


def wait_for_tasks(jobs: list[tuple[str, str]]) -> None:
    """Polls ingestion tasks and shows one progress bar per file."""
    bars = {task_id: (name, st.progress(0, text=f"{name}: queued")) for name, task_id in jobs}
    pending = set(bars)
    while pending:
        for task_id in list(pending):
            name, bar = bars[task_id]
            try:
                status = api("GET", f"/api/tasks/{task_id}", timeout=10)
            except ApiError as exc:
                bar.progress(0, text=f"{name}: {exc}")
                pending.discard(task_id)
                continue
            state, stage = status["state"], status.get("stage") or ""
            if state == "SUCCESS":
                bar.progress(100, text=f"✅ {name}: indexed")
                pending.discard(task_id)
            elif state == "FAILURE":
                bar.progress(100, text=f"❌ {name}: {status.get('result')}")
                pending.discard(task_id)
            else:
                bar.progress(int(status.get("progress") or 0), text=f"{name}: {stage or state.lower()}")
        if pending:
            time.sleep(1)


# ====================================================================== login
if "token" not in st.session_state:
    st.title("🗂️ Multimodal Asset Library")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            try:
                data = api("POST", "/api/token", data={"username": username, "password": password})
                st.session_state.update(token=data["access_token"], role=data["role"], username=username)
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
    st.stop()

is_admin = st.session_state["role"] == "admin"

# ====================================================================== sidebar
with st.sidebar:
    st.markdown(f"Signed in as **{st.session_state['username']}** (`{st.session_state['role']}`)")
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.subheader("System status")
    try:
        res = requests.get(f"{API}/ready", timeout=5)
        checks = res.json().get("checks", {})
    except (requests.RequestException, ValueError):
        checks = {"gateway": "unreachable"}
    for name, state in checks.items():
        icon = "🟢" if state == "ok" else ("🟡" if state in ("warming", "unavailable") else "🔴")
        st.markdown(f"{icon} **{name}**: {state}")
    if is_admin:
        st.divider()
        st.subheader("Admin")
        confirm = st.checkbox("I understand this deletes every asset")
        if st.button("Reset library", type="primary", disabled=not confirm, use_container_width=True):
            try:
                result = api("POST", "/api/admin/reset", timeout=120)
                st.success(f"Library reset ({result['objects_deleted']} objects removed).")
            except ApiError as exc:
                st.error(str(exc))

tab_names = ["🔍 Search", "💬 Ask", "📚 Library"] + (["📥 Upload"] if is_admin else [])
tabs = st.tabs(tab_names)

# ====================================================================== search
with tabs[0]:
    render_player("search")
    col_q, col_type, col_k = st.columns([5, 2, 1])
    query = col_q.text_input("Search across documents, transcripts, images and video frames", key="search_q")
    media_filter = col_type.selectbox("Type", ["all", "document", "image", "video", "audio"], key="search_type")
    limit = col_k.number_input("Results", 1, 20, 6, key="search_k")
    if query.strip():
        params = {"query": query, "limit": limit}
        if media_filter != "all":
            params["media_type"] = media_filter
        try:
            started = time.perf_counter()
            results = api("GET", "/api/search", params=params)
            st.caption(f"{(time.perf_counter() - started) * 1000:.0f} ms")
            left, right = st.columns(2)
            with left:
                st.markdown("#### Text matches")
                st.caption("PDF chunks, speech transcripts and image descriptions (MiniLM)")
                for i, hit in enumerate(results["text_matches"]):
                    with st.container(border=True):
                        st.markdown(
                            f"{TYPE_ICONS.get(hit['media_type'], '')} **{hit['filename']}** · {location(hit)}"
                            f" · score {hit['score']:.2f}"
                        )
                        st.write(hit["text"])
                        jump_button(hit, key=f"t{i}")
                if not results["text_matches"]:
                    st.caption("No text matches.")
            with right:
                st.markdown("#### Visual matches")
                st.caption("Images and sampled video frames (CLIP)")
                for i, hit in enumerate(results["visual_matches"]):
                    with st.container(border=True):
                        st.markdown(
                            f"{TYPE_ICONS.get(hit['media_type'], '')} **{hit['filename']}** · {location(hit)}"
                            f" · score {hit['score']:.2f}"
                        )
                        if hit["media_type"] == "image" and (
                            url := media_url(hit["asset_id"], st.session_state["token"])
                        ):
                            st.image(url, width=280)
                        jump_button(hit, key=f"v{i}")
                if not results["visual_matches"]:
                    st.caption("No visual matches.")
        except ApiError as exc:
            st.error(str(exc))

# ====================================================================== RAG
with tabs[1]:
    render_player("ask")
    st.caption("Answers are generated by a local LLM from the top matching text chunks, with numbered citations.")
    with st.form("ask"):
        question = st.text_input("Question")
        c1, c2 = st.columns(2)
        top_k = c1.slider("Sources to retrieve", 1, 10, 5)
        rag_type = c2.selectbox("Limit to type", ["all", "document", "video", "audio", "image"])
        asked = st.form_submit_button("Ask", type="primary")
    if asked and question.strip():
        body = {"question": question, "top_k": top_k}
        if rag_type != "all":
            body["media_type"] = rag_type
        with st.spinner("Retrieving sources and generating an answer..."):
            try:
                st.session_state["rag"] = api("POST", "/api/rag", json=body, timeout=240)
            except ApiError as exc:
                st.error(str(exc))
    if rag := st.session_state.get("rag"):
        st.markdown("#### Answer")
        st.markdown(rag["answer"])
        if rag["sources"]:
            st.markdown("#### Sources")
            for src in rag["sources"]:
                cited = "✅ cited" if src["n"] in rag["cited"] else "not cited"
                with st.container(border=True):
                    st.markdown(f"**[{src['n']}] {src['label']}** · score {src['score']:.2f} · {cited}")
                    st.caption(src["text"])
                    jump_button(src, key=f"src{src['n']}")

# ====================================================================== library
with tabs[2]:
    c1, c2 = st.columns([3, 1])
    lib_type = c1.selectbox("Filter", ["all", "document", "image", "video", "audio"], key="lib_type")
    if c2.button("Refresh", use_container_width=True):
        st.rerun()
    try:
        params = {"limit": 200} | ({"media_type": lib_type} if lib_type != "all" else {})
        listing = api("GET", "/api/assets", params=params)
    except ApiError as exc:
        st.error(str(exc))
        listing = {"total": 0, "assets": []}

    assets = listing["assets"]
    st.caption(f"{listing['total']} assets")
    if assets:
        st.dataframe(
            [
                {
                    "": TYPE_ICONS.get(a["media_type"], ""),
                    "file": a["filename"],
                    "status": a["status"],
                    "size": fmt_size(a["size_bytes"]),
                    "duration": fmt_ts(a["duration_seconds"]) if a["duration_seconds"] else "",
                    "indexed": ", ".join(
                        f"{v} {k.replace('_indexed', '').replace('_', ' ')}"
                        for k, v in (a["stats"] or {}).items()
                        if k in ("chunks_indexed", "frames_indexed", "transcript_windows", "image_vectors")
                    ),
                    "processing": f"{a['stats']['processing_seconds']}s"
                    if a["stats"].get("processing_seconds")
                    else "",
                    "uploaded by": a["uploaded_by"],
                }
                for a in assets
            ],
            use_container_width=True,
            hide_index=True,
        )
        by_label = {f"{TYPE_ICONS.get(a['media_type'], '')} {a['filename']} ({a['id'][:8]})": a for a in assets}
        asset = by_label[st.selectbox("Asset details", list(by_label))]

        with st.container(border=True):
            info, preview = st.columns([1, 1])
            with info:
                st.markdown(f"### {asset['filename']}")
                st.markdown(
                    f"**Status:** {asset['status']}  \n**Type:** {asset['media_type']}  \n"
                    f"**Size:** {fmt_size(asset['size_bytes'])}  \n**Uploaded:** {asset['created_at'][:19]}"
                )
                if asset["error"]:
                    st.error(asset["error"])
                if asset["caption"]:
                    st.info(f"Generated description: {asset['caption']}")
                with st.expander("Ingestion stats"):
                    st.json(asset["stats"])
            with preview:
                url = media_url(asset["id"], st.session_state["token"]) if asset["status"] == "ready" else None
                if url:
                    if asset["media_type"] == "video":
                        st.video(url)
                    elif asset["media_type"] == "audio":
                        st.audio(url)
                    elif asset["media_type"] == "image":
                        st.image(url, use_column_width=True)
                    else:
                        st.link_button("Open PDF", url)

            actions = st.columns(3 if is_admin else 1)
            summary_key = f"summary-{asset['id']}"
            if actions[0].button("Summarize", disabled=asset["status"] != "ready", use_container_width=True):
                with st.spinner("Summarizing (long assets are summarized in parts, then combined)..."):
                    try:
                        st.session_state[summary_key] = api("POST", f"/api/assets/{asset['id']}/summary", timeout=600)
                    except ApiError as exc:
                        st.error(str(exc))
            if is_admin:
                if actions[1].button(
                    "Re-index", use_container_width=True, disabled=asset["status"] not in ("ready", "failed")
                ):
                    try:
                        job = api("POST", f"/api/assets/{asset['id']}/reindex")
                        wait_for_tasks([(asset["filename"], job["task_id"])])
                    except ApiError as exc:
                        st.error(str(exc))
                if actions[2].button("Delete", use_container_width=True):
                    try:
                        api("DELETE", f"/api/assets/{asset['id']}")
                        st.success("Deleted.")
                        st.rerun()
                    except ApiError as exc:
                        st.error(str(exc))

            if summary := st.session_state.get(summary_key):
                st.markdown("#### Summary")
                st.caption(
                    f"{summary['strategy']} · {summary['llm_calls']} LLM call(s) · "
                    f"{summary['source_chunks']} source chunks{' · cached' if summary.get('cached') else ''}"
                )
                st.markdown(summary["summary"])
                st.download_button(
                    "Download as Markdown",
                    data=f"# Summary: {asset['filename']}\n\n{summary['summary']}\n",
                    file_name=f"{os.path.splitext(asset['filename'])[0]}_summary.md",
                    mime="text/markdown",
                )

# ====================================================================== upload
if is_admin:
    with tabs[3]:
        st.markdown(
            f"Upload up to **{MAX_UPLOAD_MB} MB** per file here. For larger files use the direct uploader, "
            "which sends the file straight to object storage in parts:"
        )
        st.code("DAM_PASSWORD=... python scripts/dam_upload.py path/to/large_video.mp4", language="bash")
        files = st.file_uploader("Files", type=UPLOAD_TYPES, accept_multiple_files=True)
        if files and st.button(f"Upload and index {len(files)} file(s)", type="primary"):
            jobs = []
            for f in files:
                try:
                    res = api("POST", "/api/upload", timeout=600, files={"file": (f.name, f, f.type)})
                    jobs.append((res["filename"], res["task_id"]))
                except ApiError as exc:
                    st.error(f"{f.name}: {exc}")
            if jobs:
                wait_for_tasks(jobs)
