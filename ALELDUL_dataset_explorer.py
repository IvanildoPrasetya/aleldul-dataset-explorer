import json
import html
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px

LABELS = [
    "FACILITY_POS", "FACILITY_NEG",
    "CLEANLINESS_POS", "CLEANLINESS_NEG",
    "LINEN_POS", "LINEN_NEG",
    "SERVICE_POS", "SERVICE_NEG",
    "ODOR_POS", "ODOR_NEG",
    "ROOM_CONDITION_POS", "ROOM_CONDITION_NEG",
    "BREAKFAST_POS", "BREAKFAST_NEG",
]

ASPECTS = [
    "FACILITY", "CLEANLINESS", "LINEN", "SERVICE",
    "ODOR", "ROOM_CONDITION", "BREAKFAST"
]

st.set_page_config(
    page_title="ALELDUL HoASA Dataset Explorer",
    page_icon="🏨",
    layout="wide"
)

st.title("ALELDUL HoASA — Dataset Explorer")
st.caption(
    "Hotel Aspect-based Sentiment Analysis | "
    "Eksplorasi anotasi multilabel (Schema A) dan span NER (Schema B)"
)


@st.cache_data(show_spinner=False)
def load_jsonl(file_bytes):
    records = []
    content = file_bytes.decode("utf-8-sig")
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON tidak valid pada baris {line_number}: {exc}"
            )
    return records


def normalize_record(record, index):
    """Normalisasi ringan; tidak mengubah isi teks atau anotasi."""
    return {
        "id": record.get("id", index + 1),
        "text": str(record.get("text", "")),
        "labels": record.get("labels") or [],
        "entities": record.get("entities") or [],
        "flag": record.get("flag")
    }


def make_dataframe(records):
    rows = []
    for i, record in enumerate(records):
        item = normalize_record(record, i)
        rows.append({
            "id": item["id"],
            "text": item["text"],
            "labels": item["labels"],
            "entities": item["entities"],
            "flag": item["flag"],
            "text_length": len(item["text"]),
            "label_count": len(item["labels"]),
            "entity_count": len(item["entities"]),
            "needs_review": item["flag"] == "needs_review"
        })
    return pd.DataFrame(rows)


def highlight_spans(text, entities):
    """Highlight span menggunakan offset karakter [start, end)."""
    valid_entities = []
    for entity in entities:
        try:
            start = int(entity["start"])
            end = int(entity["end"])
            label = str(entity["label"])
            span_text = str(entity["text"])
            if 0 <= start < end <= len(text) and text[start:end] == span_text:
                valid_entities.append((start, end, label, span_text))
        except (KeyError, TypeError, ValueError):
            continue

    valid_entities.sort(key=lambda x: (x[0], x[1]))
    parts = []
    cursor = 0
    for start, end, label, span_text in valid_entities:
        if start < cursor:
            continue
        parts.append(html.escape(text[cursor:start]))
        parts.append(
            '<mark style="background-color:#ffe08a; padding:2px 4px; '
            'border-radius:4px;" title="' + html.escape(label) + '">'
            + html.escape(text[start:end])
            + '</mark>'
            '<small style="color:#555;">[' + html.escape(label) + ']</small>'
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def entity_table(records):
    rows = []
    for record in records:
        for entity in record["entities"]:
            rows.append({
                "review_id": record["id"],
                "span": entity.get("text", ""),
                "label": entity.get("label", ""),
                "start": entity.get("start", None),
                "end": entity.get("end", None),
                "text": record["text"]
            })
    return pd.DataFrame(rows)


st.sidebar.header("📂 Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file gold adjudication (.jsonl)",
    type=["jsonl"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
else:
    st.sidebar.info(
        "Unggah hoasa_gold_adjudication.jsonl untuk mulai. "
        "File tidak dikirim ke server eksternal."
    )
    st.stop()

try:
    raw_records = load_jsonl(file_bytes)
except Exception as error:
    st.error(f"Gagal membaca dataset: {error}")
    st.stop()

records = [normalize_record(r, i) for i, r in enumerate(raw_records)]
df = make_dataframe(raw_records)
entities_df = entity_table(records)

if df.empty:
    st.warning("Dataset tidak memiliki baris data.")
    st.stop()

st.sidebar.divider()
page = st.sidebar.radio(
    "Navigasi",
    [
        "Overview",
        "Browse Data",
        "Multilabel Analysis",
        "NER Analysis",
        "Filter/Search"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Filter global")
selected_labels = st.sidebar.multiselect(
    "Label Schema A",
    options=LABELS,
    help="Pilih satu atau beberapa label untuk menyaring review."
)
only_review = st.sidebar.checkbox("Hanya needs_review", value=False)

filtered_df = df.copy()
if selected_labels:
    filtered_df = filtered_df[
        filtered_df["labels"].apply(
            lambda labels: any(label in labels for label in selected_labels)
        )
    ]
if only_review:
    filtered_df = filtered_df[filtered_df["needs_review"]]

filtered_ids = set(filtered_df["id"].tolist())
filtered_records = [r for r in records if r["id"] in filtered_ids]
filtered_entities_df = entity_table(filtered_records)

if page == "Overview":
    st.header("Overview Dataset")

    total_reviews = len(df)
    total_entities = sum(len(r["entities"]) for r in records)
    total_labels = sum(len(r["labels"]) for r in records)
    review_flag_count = int(df["needs_review"].sum())
    empty_label_count = int((df["label_count"] == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Review", f"{total_reviews:,}")
    c2.metric("Total Span / Entity", f"{total_entities:,}")
    c3.metric("Total Label", f"{total_labels:,}")
    c4.metric("Needs Review", f"{review_flag_count:,}")

    st.subheader("Ringkasan Dataset")
    st.write(
        "Dataset yang digunakan adalah gold adjudication HoASA. "
        "Setiap review memiliki anotasi Schema A berupa daftar label multilabel "
        "dan Schema B berupa span teks yang menjadi bukti anotasi."
    )

    c5, c6, c7 = st.columns(3)
    c5.metric("Rata-rata panjang teks", f"{df['text_length'].mean():.1f} karakter")
    c6.metric("Rata-rata label/review", f"{df['label_count'].mean():.2f}")
    c7.metric("Review tanpa label", f"{empty_label_count:,}")

    st.subheader("Taxonomy Label")
    taxonomy_rows = []
    for aspect in ASPECTS:
        taxonomy_rows.append({
            "Aspek": aspect,
            "Positif": f"{aspect}_POS",
            "Negatif": f"{aspect}_NEG"
        })
    st.dataframe(pd.DataFrame(taxonomy_rows), use_container_width=True, hide_index=True)

    st.info(
        "Taxonomy terdiri dari 7 aspek dan 2 polaritas (POS/NEG), "
        "sehingga terdapat 14 label resmi. Label NEUTRAL tidak digunakan."
    )

    st.subheader("Distribusi Polaritas")
    polarity_counts = {
        "POS": sum(
            1 for r in records for label in r["labels"] if label.endswith("_POS")
        ),
        "NEG": sum(
            1 for r in records for label in r["labels"] if label.endswith("_NEG")
        )
    }
    polarity_df = pd.DataFrame({
        "Polaritas": list(polarity_counts.keys()),
        "Jumlah label": list(polarity_counts.values())
    })
    fig = px.bar(
        polarity_df, x="Polaritas", y="Jumlah label",
        color="Polaritas", title="Jumlah anotasi berdasarkan polaritas"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Browse Data":
    st.header("Browse Data")
    st.caption("Lihat teks review, label Schema A, dan span Schema B.")

    page_size = st.selectbox("Jumlah review per halaman", [10, 20, 50, 100])
    total = len(filtered_records)
    if total == 0:
        st.warning("Tidak ada review yang sesuai dengan filter.")
    else:
        max_page = max(1, (total - 1) // page_size + 1)
        page_number = st.number_input(
            "Halaman", min_value=1, max_value=max_page, value=1, step=1
        )
        start_idx = (page_number - 1) * page_size
        current_records = filtered_records[start_idx:start_idx + page_size]
        st.write(f"Menampilkan {start_idx + 1}–{start_idx + len(current_records)} dari {total} review.")

        for record in current_records:
            with st.expander(
                f"Review ID {record['id']} | "
                f"{len(record['labels'])} label | "
                f"{len(record['entities'])} span",
                expanded=False
            ):
                st.markdown("**Teks review**")
                st.write(record["text"])

                st.markdown("**Schema A — Multilabel**")
                if record["labels"]:
                    st.write(", ".join(record["labels"]))
                else:
                    st.write("Tidak ada label")

                st.markdown("**Schema B — Span NER (highlight)**")
                highlighted = highlight_spans(record["text"], record["entities"])
                st.markdown(
                    '<div style="line-height:2; padding:12px; border:1px solid #ddd; '
                    'border-radius:8px;">' + highlighted + '</div>',
                    unsafe_allow_html=True
                )

                if record["entities"]:
                    span_rows = [{
                        "Span": e.get("text", ""),
                        "Label": e.get("label", ""),
                        "Start": e.get("start", None),
                        "End": e.get("end", None)
                    } for e in record["entities"]]
                    st.dataframe(pd.DataFrame(span_rows), use_container_width=True, hide_index=True)
                else:
                    st.write("Tidak ada span/entity.")

                st.caption(f"Flag: {record['flag'] if record['flag'] else '-'}")

elif page == "Multilabel Analysis":
    st.header("Multilabel Analysis")
    st.caption("Analisis distribusi label Schema A pada tingkat review.")

    if filtered_df.empty:
        st.warning("Tidak ada review yang sesuai dengan filter.")
    else:
        label_counts = Counter(
            label for labels in filtered_df["labels"] for label in labels
        )
        label_df = pd.DataFrame([
            {"Label": label, "Jumlah review": label_counts.get(label, 0)}
            for label in LABELS
        ])

        st.subheader("Distribusi 14 Label")
        fig = px.bar(
            label_df.sort_values("Jumlah review", ascending=False),
            x="Label", y="Jumlah review", color="Jumlah review",
            title="Frekuensi label pada tingkat review"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            label_df.sort_values("Jumlah review", ascending=False),
            use_container_width=True, hide_index=True
        )

        st.subheader("Kombinasi Label")
        combo_counts = Counter(
            " + ".join(sorted(labels)) if labels else "(Tanpa label)"
            for labels in filtered_df["labels"]
        )
        combo_df = pd.DataFrame(
            [{"Kombinasi label": k, "Jumlah review": v}
             for k, v in combo_counts.most_common(20)]
        )
        st.write("20 kombinasi label yang paling sering muncul.")
        st.dataframe(combo_df, use_container_width=True, hide_index=True)

        st.subheader("Jumlah Label per Review")
        fig2 = px.histogram(
            filtered_df, x="label_count", nbins=15,
            title="Distribusi jumlah label dalam satu review"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "Unduh hasil filter (CSV)",
            data=filtered_df[["id", "text", "labels", "flag"]].to_csv(index=False).encode("utf-8"),
            file_name="hoasa_multilabel_filtered.csv",
            mime="text/csv"
        )

elif page == "NER Analysis":
    st.header("NER Analysis")
    st.caption("Analisis distribusi entity/span pada tingkat teks (Schema B).")

    if filtered_entities_df.empty:
        st.warning("Tidak ada span yang sesuai dengan filter.")
    else:
        entity_counts = (
            filtered_entities_df["label"].value_counts()
            .reindex(LABELS, fill_value=0)
            .rename_axis("Label")
            .reset_index(name="Jumlah span")
        )

        c1, c2 = st.columns(2)
        c1.metric("Total span pada hasil filter", len(filtered_entities_df))
        c2.metric("Label span unik", filtered_entities_df["label"].nunique())

        st.subheader("Distribusi Span per Label")
        fig = px.bar(
            entity_counts.sort_values("Jumlah span", ascending=False),
            x="Label", y="Jumlah span", color="Jumlah span",
            title="Frekuensi entity berdasarkan label"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Panjang Span")
        filtered_entities_df["panjang_karakter"] = (
            filtered_entities_df["end"] - filtered_entities_df["start"]
        )
        fig2 = px.histogram(
            filtered_entities_df, x="panjang_karakter",
            title="Distribusi panjang span (karakter)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Contoh Span yang Di-highlight")
        examples = filtered_entities_df.head(20)
        for _, row in examples.iterrows():
            with st.expander(f"Review ID {row['review_id']} — {row['label']}"):
                st.markdown(
                    '<div style="line-height:2; padding:12px; border:1px solid #ddd; '
                    'border-radius:8px;">'
                    + highlight_spans(row["text"], [
                        {"text": row["span"], "label": row["label"],
                         "start": row["start"], "end": row["end"]}
                    ])
                    + '</div>',
                    unsafe_allow_html=True
                )
                st.write(f"Span: {row['span']}")
                st.write(f"Offset: {row['start']}–{row['end']}")

        st.download_button(
            "Unduh daftar span (CSV)",
            data=filtered_entities_df.drop(columns=["text"]).to_csv(index=False).encode("utf-8"),
            file_name="hoasa_ner_spans.csv",
            mime="text/csv"
        )

elif page == "Filter/Search":
    st.header("Filter / Search")
    st.caption("Cari kata atau frasa pada teks review, lalu gabungkan dengan filter label/entity.")

    query = st.text_input("Kata atau frasa yang dicari", placeholder="Contoh: AC tidak dingin")
    entity_label_filter = st.multiselect(
        "Filter berdasarkan label entity (Schema B)",
        options=LABELS
    )
    entity_text_query = st.text_input(
        "Cari teks span/entity (opsional)",
        placeholder="Contoh: kamar mandi kotor"
    )

    result_records = filtered_records

    if query.strip():
        q = query.strip().casefold()
        result_records = [
            r for r in result_records if q in r["text"].casefold()
        ]

    if entity_label_filter:
        result_records = [
            r for r in result_records
            if any(e.get("label") in entity_label_filter for e in r["entities"])
        ]

    if entity_text_query.strip():
        eq = entity_text_query.strip().casefold()
        result_records = [
            r for r in result_records
            if any(eq in str(e.get("text", "")).casefold() for e in r["entities"])
        ]

    st.write(f"**Hasil ditemukan: {len(result_records)} review**")

    for record in result_records[:100]:
        with st.expander(
            f"Review ID {record['id']} | {len(record['labels'])} label | {len(record['entities'])} span"
        ):
            st.write(record["text"])
            st.markdown("**Label:** " + (", ".join(record["labels"]) if record["labels"] else "Tanpa label"))
            st.markdown(
                '<div style="line-height:2; padding:12px; border:1px solid #ddd; '
                'border-radius:8px;">'
                + highlight_spans(record["text"], record["entities"])
                + '</div>',
                unsafe_allow_html=True
            )

    if len(result_records) > 100:
        st.info("Hanya 100 hasil pertama yang ditampilkan. Persempit pencarian untuk melihat hasil yang lebih spesifik.")

    if result_records:
        export_rows = [{
            "id": r["id"],
            "text": r["text"],
            "labels": "|".join(r["labels"]),
            "entities": json.dumps(r["entities"], ensure_ascii=False),
            "flag": r["flag"]
        } for r in result_records]
        st.download_button(
            "Unduh hasil pencarian (CSV)",
            data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
            file_name="hoasa_search_results.csv",
            mime="text/csv"
        )

st.sidebar.divider()
st.sidebar.caption(
    "ALELDUL | HoASA UTS Dataset Explorer\n\n"
    "Aplikasi ini hanya untuk eksplorasi dataset dan anotasi, bukan prediksi sentimen."
)
