import io
import json

import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="HL7v2/FHIR to OMOP/Tuva", page_icon="🏥", layout="wide")

session = get_active_session()

st.title("HL7v2/FHIR to OMOP/Tuva Transformer")
st.caption(
    "Vocabulary seeds from [Tuva Health](https://thetuvaproject.com/) (Apache 2.0) + "
    "[OHDSI Athena](https://athena.ohdsi.org/) · OMOP CDM v5.4 + Tuva Input Layer · "
    "HL7v2 + FHIR R4 (JSON & XML) auto-detected · "
    "Multi-cloud · Your data never leaves your account"
)


@st.cache_data(ttl=300)
def list_databases():
    try:
        return [r['name'] for r in session.sql("SHOW DATABASES").collect()]
    except:
        return []


@st.cache_data(ttl=120)
def list_schemas(db):
    try:
        return [r['name'] for r in session.sql(f"SHOW SCHEMAS IN DATABASE {db}").collect()]
    except:
        return []


@st.cache_data(ttl=60)
def list_tables(db, schema):
    try:
        rows = session.sql(f"SHOW TABLES IN {db}.{schema}").collect()
        return [r['name'] for r in rows]
    except:
        return []


@st.cache_data(ttl=60)
def list_columns(db, schema, table):
    try:
        rows = session.sql(f"DESCRIBE TABLE {db}.{schema}.{table}").collect()
        return [r['name'] for r in rows]
    except:
        return []


@st.cache_data(ttl=60)
def list_columns_with_types(db, schema, table):
    try:
        rows = session.sql(f"DESCRIBE TABLE {db}.{schema}.{table}").collect()
        return [(r['name'], str(r['type']).upper()) for r in rows]
    except:
        return []


@st.cache_data(ttl=30)
def detect_source_format(db, schema, table, data_col):
    try:
        row = session.sql(f"""
            SELECT SUBSTR(TRIM({data_col}::VARCHAR), 1, 4) AS first_chars
            FROM {db}.{schema}.{table}
            LIMIT 1
        """).collect()
        if not row:
            return 'unknown'
        fc = row[0]['FIRST_CHARS'] or ''
        if fc[:4] == 'MSH|' or fc[:3] == 'MSH':
            return 'hl7v2'
        if fc[:1] == '<':
            return 'xml'
        if fc[:1] == '{':
            return 'json'
        return 'unknown'
    except:
        return 'unknown'


@st.cache_data(ttl=60)
def get_table_preview(db, schema, table, limit=5):
    try:
        return session.sql(f"SELECT * FROM {db}.{schema}.{table} LIMIT {limit}").to_pandas()
    except:
        return pd.DataFrame()


def load_config():
    try:
        rows = session.sql("SELECT key, value FROM app_state.configuration").collect()
        return {r['KEY']: r['VALUE'] for r in rows}
    except:
        return {}


PROTECTED_SCHEMAS = {'OMOP_CDM'}

tab_docs, tab_config, tab_run, tab_history, tab_vocab, tab_coverage, tab_quality, tab_explore = st.tabs([
    "📄 Docs", "Configure", "Run", "History",
    "Vocabulary", "Coverage", "Quality", "Explore",
])

with tab_docs:
    st.subheader("Clinical Document Intelligence")
    st.caption("Upload clinical notes or PDFs → AI extracts structured data → Review & correct → Generate FHIR → Ingest to OMOP")

    DOC_DB = "HEALTHCARE_DOC_AI"
    DOC_SCHEMA = "DOC_INTELLIGENCE"
    DOC_STAGE = f"@{DOC_DB}.{DOC_SCHEMA}.RAW_DOCUMENTS"

    doc_tab_upload, doc_tab_review, doc_tab_history = st.tabs(["Upload & Extract", "Review & Correct", "Extraction History"])

    with doc_tab_upload:
        st.markdown("#### Upload a Clinical Note")
        upload_method = st.radio("Input method", ["Select existing", "Paste text", "Upload PDF"], horizontal=True, key="doc_input_method")

        note_text = ""
        if upload_method == "Select existing":
            try:
                existing_notes = session.sql(f"""
                    SELECT note_id, patient_mrn, note_type, department,
                           LEFT(note_text, 60) || '...' as preview, note_text
                    FROM {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
                    ORDER BY created_at DESC
                """).to_pandas()
                if not existing_notes.empty:
                    selected_idx = st.selectbox("Select a note",
                        range(len(existing_notes)),
                        format_func=lambda i: f"#{existing_notes.iloc[i]['NOTE_ID']} — {existing_notes.iloc[i]['NOTE_TYPE']} ({existing_notes.iloc[i]['DEPARTMENT']}) — {existing_notes.iloc[i]['PREVIEW']}",
                        key="doc_existing_select")
                    note_text = existing_notes.iloc[selected_idx]['NOTE_TEXT']
                    st.text_area("Note content", value=note_text, height=250, disabled=True, key="doc_existing_preview")
                else:
                    st.info("No notes in database yet. Upload or paste one first!")
            except Exception as e:
                st.warning(f"Could not load notes: {str(e)[:200]}")

        elif upload_method == "Paste text":
            note_text = st.text_area("Paste clinical note text", height=250, key="doc_paste_text",
                placeholder="4mo M s/p CAVC repair POD2, doing well o/n per RN...")
            note_type = st.selectbox("Note type", ["Progress Note", "Discharge Summary", "H&P", "Clinic Note", "Consult", "Op Note"], key="doc_note_type")
            note_dept = st.text_input("Department", value="", key="doc_dept", placeholder="e.g., Cardiac ICU, Oncology, NICU")

        if upload_method in ("Select existing", "Paste text") and note_text:
            if st.button("Extract", key="doc_extract_btn", type="primary"):
                with st.spinner("Running AI extraction..."):
                    try:
                        escaped = note_text.replace("'", "''")
                        prompt = f"""Extract ALL clinical observations from this note as structured JSON. For EVERY numeric value mentioned, capture it.

Return JSON with these arrays:
- "observations": array of {{"name": "human-readable name", "value": numeric_value_or_string, "unit": "unit", "category": "vital-signs|laboratory|social-history"}}
- "diagnoses": array of {{"name": "diagnosis", "icd10": "code if identifiable"}}
- "medications": array of {{"drug": "name", "dose": "dose", "route": "route", "frequency": "frequency"}}
- "procedures": array of strings

Rules:
- Blood pressure: split into TWO observations (systolic + diastolic)
- Include ALL numeric values: weights, heights, BMI, gestational age measurements, fetal heart rate, urine dipstick results
- For qualitative results (e.g., "2+ protein"), use the qualitative value as a string
- Do NOT skip values just because they appear in flowing text
- Do NOT invent values not present in the note

Clinical note:
{escaped}"""
                        prompt_escaped = prompt.replace("'", "''")
                        result = session.sql(f"""
                            SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', '{prompt_escaped}')::STRING as extracted
                        """).collect()
                        if result:
                            raw = result[0]['EXTRACTED']
                            try:
                                resp = json.loads(raw)
                            except json.JSONDecodeError:
                                import re
                                json_match = re.search(r'\{[\s\S]*\}', raw)
                                resp = json.loads(json_match.group()) if json_match else {"error": "Could not parse"}

                            st.success("Extraction complete!")

                            obs = resp.get('observations', [])
                            dx = resp.get('diagnoses', [])
                            meds = resp.get('medications', [])
                            procs = resp.get('procedures', [])

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Observations (Vitals + Labs)**")
                                if obs:
                                    for o in obs:
                                        val = o.get('value', '')
                                        unit = o.get('unit', '')
                                        cat = o.get('category', '')
                                        st.write(f"• **{o.get('name', '?')}**: {val} {unit} _({cat})_")
                                else:
                                    st.write("_None identified_")

                                st.markdown("**Diagnoses**")
                                if dx:
                                    for d in dx:
                                        icd = f" [{d.get('icd10')}]" if d.get('icd10') else ""
                                        st.write(f"• {d.get('name', '?')}{icd}")
                                else:
                                    st.write("_None identified_")

                            with col2:
                                st.markdown("**Medications**")
                                if meds:
                                    for m in meds:
                                        st.write(f"• {m.get('drug', '?')} {m.get('dose', '')} {m.get('route', '')} {m.get('frequency', '')}")
                                else:
                                    st.write("_None identified_")

                                st.markdown("**Procedures**")
                                if procs:
                                    for p in procs:
                                        st.write(f"• {p}")
                                else:
                                    st.write("_None identified_")

                            st.markdown("---")

                            if obs or dx or meds or procs:
                                st.markdown("---")
                                st.markdown("### Terminology Mapping & FHIR Generation")

                                loinc_results = session.sql("""
                                    SELECT LOINC_CODE, LOINC_DESCRIPTION FROM TRE_HEALTHCARE_DB.TUVA_TEST_OMOP.LOINC_TO_OMOP
                                """).to_pandas()
                                rxnorm_results = session.sql("""
                                    SELECT CONCEPT_CODE, CONCEPT_NAME FROM TUVA_FHIR_TO_OMOP_APP.TERMINOLOGY.CONCEPT
                                    WHERE VOCABULARY_ID = 'RxNorm'
                                """).to_pandas()
                                icd10_results = session.sql("""
                                    SELECT CONCEPT_CODE, CONCEPT_NAME FROM TUVA_FHIR_TO_OMOP_APP.TERMINOLOGY.CONCEPT
                                    WHERE VOCABULARY_ID = 'ICD10CM' AND DOMAIN_ID = 'Condition'
                                """).to_pandas()
                                snomed_results = session.sql("""
                                    SELECT CONCEPT_CODE, CONCEPT_NAME FROM TUVA_FHIR_TO_OMOP_APP.TERMINOLOGY.CONCEPT
                                    WHERE VOCABULARY_ID = 'SNOMED' AND DOMAIN_ID = 'Condition'
                                """).to_pandas()

                                loinc_lookup = {row['LOINC_DESCRIPTION'].lower(): row['LOINC_CODE'] for _, row in loinc_results.iterrows()}
                                rxnorm_lookup = {row['CONCEPT_NAME'].lower(): row['CONCEPT_CODE'] for _, row in rxnorm_results.iterrows()}
                                icd10_lookup = {row['CONCEPT_NAME'].lower(): row['CONCEPT_CODE'] for _, row in icd10_results.iterrows()}
                                snomed_lookup = {row['CONCEPT_NAME'].lower(): row['CONCEPT_CODE'] for _, row in snomed_results.iterrows()}

                                name_to_loinc = {
                                    "systolic blood pressure": "8480-6", "diastolic blood pressure": "8462-4",
                                    "heart rate": "8867-4", "body temperature": "8310-5",
                                    "respiratory rate": "9279-1", "oxygen saturation": "2708-6",
                                    "body weight": "29463-7", "body height": "8302-2",
                                    "body mass index": "39156-5", "bmi": "39156-5",
                                    "fetal heart rate": "55283-6", "fundal height": "11881-0",
                                    "hemoglobin": "718-7", "hgb": "718-7",
                                    "platelets": "777-3", "plts": "777-3",
                                    "inr": "6301-6", "glucose": "2345-7",
                                    "creatinine": "2160-0", "potassium": "2823-3",
                                    "sodium": "2951-2", "protein in urine": "2514-8",
                                    "urine protein": "2514-8", "white blood cell count": "6690-2",
                                    "wbc": "6690-2", "weight": "29463-7",
                                }

                                fhir_bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}

                                obs_mapped = 0
                                if obs:
                                    for o in obs:
                                        name_lower = o.get('name', '').lower()
                                        loinc_code = name_to_loinc.get(name_lower)
                                        if not loinc_code:
                                            for key, code in name_to_loinc.items():
                                                if key in name_lower or name_lower in key:
                                                    loinc_code = code
                                                    break
                                        if not loinc_code:
                                            for desc, code in loinc_lookup.items():
                                                if name_lower in desc or desc in name_lower:
                                                    loinc_code = code
                                                    break
                                        if loinc_code:
                                            obs_mapped += 1
                                        fhir_bundle["entry"].append({"resource": {
                                            "resourceType": "Observation",
                                            "status": "final",
                                            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": o.get('category', 'vital-signs')}]}],
                                            "code": {"coding": [{"system": "http://loinc.org", "code": loinc_code or "UNMAPPED", "display": o.get('name', '')}]},
                                            "valueQuantity": {"value": o.get('value'), "unit": o.get('unit', '')},
                                        }})

                                meds_mapped = 0
                                if meds:
                                    for m in meds:
                                        drug_lower = m.get('drug', '').lower()
                                        rxnorm_code = rxnorm_lookup.get(drug_lower)
                                        if not rxnorm_code:
                                            for name, code in rxnorm_lookup.items():
                                                if name in drug_lower or drug_lower in name:
                                                    rxnorm_code = code
                                                    break
                                        if rxnorm_code:
                                            meds_mapped += 1
                                        fhir_bundle["entry"].append({"resource": {
                                            "resourceType": "MedicationStatement",
                                            "status": "active",
                                            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": rxnorm_code or "UNMAPPED", "display": m.get('drug', '')}]},
                                            "dosage": [{"text": f"{m.get('dose', '')} {m.get('route', '')} {m.get('frequency', '')}".strip()}],
                                        }})

                                dx_mapped = 0
                                if dx:
                                    for d in dx:
                                        dx_name = d.get('name', '').lower()
                                        icd10_code = d.get('icd10')
                                        snomed_code = None
                                        if not icd10_code:
                                            for name, code in icd10_lookup.items():
                                                if dx_name in name or name in dx_name:
                                                    icd10_code = code
                                                    break
                                        for name, code in snomed_lookup.items():
                                            if dx_name in name or name in dx_name:
                                                snomed_code = code
                                                break
                                        if icd10_code or snomed_code:
                                            dx_mapped += 1
                                        coding = []
                                        if icd10_code:
                                            coding.append({"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": icd10_code, "display": d.get('name', '')})
                                        if snomed_code:
                                            coding.append({"system": "http://snomed.info/sct", "code": snomed_code, "display": d.get('name', '')})
                                        if not coding:
                                            coding.append({"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "UNMAPPED", "display": d.get('name', '')})
                                        fhir_bundle["entry"].append({"resource": {
                                            "resourceType": "Condition",
                                            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                                            "code": {"coding": coding},
                                        }})

                                if procs:
                                    cpt_results = session.sql("""
                                        SELECT CONCEPT_CODE, CONCEPT_NAME FROM TRE_HEALTHCARE_DB.TUVA_TEST_OMOP.CONCEPT
                                        WHERE VOCABULARY_ID = 'CPT4'
                                    """).to_pandas()
                                    cpt_lookup = {row['CONCEPT_NAME'].lower(): row['CONCEPT_CODE'] for _, row in cpt_results.iterrows()}
                                    procs_mapped = 0
                                    for p in procs:
                                        p_lower = p.lower() if isinstance(p, str) else str(p).lower()
                                        cpt_code = cpt_lookup.get(p_lower)
                                        if not cpt_code:
                                            for name, code in cpt_lookup.items():
                                                if p_lower in name or name in p_lower:
                                                    cpt_code = code
                                                    break
                                        if cpt_code:
                                            procs_mapped += 1
                                        coding = [{"system": "http://www.ama-assn.org/go/cpt", "code": cpt_code or "UNMAPPED", "display": p}]
                                        fhir_bundle["entry"].append({"resource": {
                                            "resourceType": "Procedure",
                                            "status": "completed",
                                            "code": {"coding": coding},
                                        }})

                                total_resources = len(fhir_bundle["entry"])
                                procs_mapped = procs_mapped if procs else 0
                                total_mapped = obs_mapped + meds_mapped + dx_mapped + procs_mapped
                                total_mappable = len(obs) + len(meds) + len(dx) + len(procs)

                                c1, c2, c3, c4, c5 = st.columns(5)
                                c1.metric("FHIR Resources", total_resources)
                                c2.metric("LOINC", f"{obs_mapped}/{len(obs)}" if obs else "0/0")
                                c3.metric("RxNorm", f"{meds_mapped}/{len(meds)}" if meds else "0/0")
                                c4.metric("ICD-10/SNOMED", f"{dx_mapped}/{len(dx)}" if dx else "0/0")
                                c5.metric("CPT", f"{procs_mapped}/{len(procs)}" if procs else "0/0")

                                if total_mappable > 0:
                                    pct = int(total_mapped / total_mappable * 100)
                                    if pct == 100:
                                        st.success(f"100% terminology coverage — all {total_mappable} items mapped to standard codes")
                                    elif pct >= 75:
                                        st.info(f"{pct}% mapped — {total_mappable - total_mapped} items need manual review")
                                    else:
                                        st.warning(f"{pct}% mapped — {total_mappable - total_mapped} unmapped items (expand vocabulary or review)")

                                with st.expander(f"FHIR Bundle ({total_resources} resources)"):
                                    st.json(fhir_bundle)

                            with st.expander("Raw extraction JSON"):
                                st.json(resp)

                            resp_json = json.dumps(resp).replace("'", "''")
                            if upload_method == "Select existing":
                                session.sql(f"""
                                    UPDATE {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
                                    SET extraction_json = PARSE_JSON('{resp_json}'),
                                        extraction_status = 'EXTRACTED'
                                    WHERE note_id = {existing_notes.iloc[selected_idx]['NOTE_ID']}
                                """).collect()
                                st.success("Extraction saved to note record!")
                            elif upload_method == "Paste text":
                                if st.button("Save to database", key="doc_save_btn"):
                                    session.sql(f"""
                                        INSERT INTO {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
                                        (patient_mrn, note_type, note_date, department, author_role, note_text, extraction_json, extraction_status)
                                        VALUES ('MANUAL-UPLOAD', '{note_type}', CURRENT_DATE(), '{note_dept}', 'Manual Upload', '{escaped}', PARSE_JSON('{resp_json}'), 'EXTRACTED')
                                    """).collect()
                                    st.success("Saved to CLINICAL_NOTES table!")
                    except Exception as e:
                        st.error(f"Extraction failed: {str(e)[:300]}")

        elif upload_method == "Upload PDF":
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="doc_pdf_upload")
            if uploaded_file:
                st.info(f"📎 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
                if st.button("Upload & Extract", key="doc_pdf_extract_btn", type="primary"):
                    with st.spinner("Uploading to stage..."):
                        try:
                            file_bytes = io.BytesIO(uploaded_file.getvalue())
                            session.file.put_stream(
                                file_bytes,
                                f"{DOC_STAGE}/{uploaded_file.name}",
                                auto_compress=False,
                                overwrite=True,
                            )
                            st.success(f"Uploaded to {DOC_STAGE}/{uploaded_file.name}")

                            with st.spinner("Running AI_PARSE_DOCUMENT..."):
                                parse_result = session.sql(f"""
                                    SELECT SNOWFLAKE.CORTEX.AI_PARSE_DOCUMENT(
                                        BUILD_SCOPED_FILE_URL('{DOC_DB}.{DOC_SCHEMA}.RAW_DOCUMENTS', '{uploaded_file.name}'),
                                        OBJECT_CONSTRUCT('mode', 'LAYOUT')
                                    )::STRING as parsed
                                """).collect()
                                if parse_result:
                                    parsed = json.loads(parse_result[0]['PARSED'])
                                    doc_text = parsed.get('content', parsed.get('text', str(parsed)))
                                    st.text_area("Extracted text", value=doc_text[:3000], height=200, disabled=True)

                                    with st.spinner("Running AI extraction on parsed text..."):
                                        escaped_doc = doc_text[:4000].replace("'", "''")
                                        extract_result = session.sql(f"""
                                            SELECT SNOWFLAKE.CORTEX.AI_EXTRACT('{escaped_doc}', OBJECT_CONSTRUCT(
                                                'diagnoses', 'List all medical diagnoses with ICD-10 codes',
                                                'medications', 'List medications with dose and frequency',
                                                'vital_signs', 'Extract vital signs with values',
                                                'procedures', 'Procedures performed or planned'
                                            ))::STRING as extracted
                                        """).collect()
                                        if extract_result:
                                            extracted = json.loads(extract_result[0]['EXTRACTED'])
                                            st.json(extracted.get('response', extracted))
                        except Exception as e:
                            st.error(f"Error: {str(e)[:300]}")

    with doc_tab_review:
        st.markdown("#### Review & Correct Extractions")
        st.caption("Review AI extractions item by item. Correct errors in names or codes. Before/after pairs are stored for quality scoring.")
        try:
            notes_df = session.sql(f"""
                SELECT note_id, patient_mrn, note_type, department, extraction_status,
                       LEFT(note_text, 80) || '...' as preview
                FROM {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
                WHERE extraction_status IN ('EXTRACTED', 'REVIEWED')
                ORDER BY created_at DESC
                LIMIT 20
            """).to_pandas()
            if not notes_df.empty:
                selected_note = st.selectbox("Select note to review",
                    range(len(notes_df)),
                    format_func=lambda i: f"#{notes_df.iloc[i]['NOTE_ID']} — {notes_df.iloc[i]['NOTE_TYPE']} ({notes_df.iloc[i]['DEPARTMENT']}) — {notes_df.iloc[i]['PREVIEW']}",
                    key="doc_review_select")

                note_id = notes_df.iloc[selected_note]['NOTE_ID']
                note_row = session.sql(f"""
                    SELECT note_text, extraction_json::STRING as extraction_json
                    FROM {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
                    WHERE note_id = {note_id}
                """).collect()

                if note_row and note_row[0]['EXTRACTION_JSON']:
                    extraction = json.loads(note_row[0]['EXTRACTION_JSON'])

                    loinc_df = session.sql("SELECT LOINC_CODE, LOINC_DESCRIPTION FROM TRE_HEALTHCARE_DB.TUVA_TEST_OMOP.LOINC_TO_OMOP").to_pandas()
                    rxnorm_df = session.sql("SELECT CONCEPT_CODE, CONCEPT_NAME FROM TUVA_FHIR_TO_OMOP_APP.TERMINOLOGY.CONCEPT WHERE VOCABULARY_ID = 'RxNorm'").to_pandas()
                    icd10_df = session.sql("SELECT CONCEPT_CODE, CONCEPT_NAME FROM TUVA_FHIR_TO_OMOP_APP.TERMINOLOGY.CONCEPT WHERE VOCABULARY_ID = 'ICD10CM' AND DOMAIN_ID = 'Condition'").to_pandas()
                    cpt_df = session.sql("SELECT CONCEPT_CODE, CONCEPT_NAME FROM TRE_HEALTHCARE_DB.TUVA_TEST_OMOP.CONCEPT WHERE VOCABULARY_ID = 'CPT4'").to_pandas()

                    loinc_lookup = {row['LOINC_DESCRIPTION'].lower(): row['LOINC_CODE'] for _, row in loinc_df.iterrows()}
                    rxnorm_lookup = {row['CONCEPT_NAME'].lower(): row['CONCEPT_CODE'] for _, row in rxnorm_df.iterrows()}
                    icd10_lookup = {row['CONCEPT_NAME'].lower(): row['CONCEPT_CODE'] for _, row in icd10_df.iterrows()}
                    cpt_lookup = {row['CONCEPT_NAME'].lower(): row['CONCEPT_CODE'] for _, row in cpt_df.iterrows()}

                    name_to_loinc = {
                        "systolic blood pressure": "8480-6", "diastolic blood pressure": "8462-4",
                        "heart rate": "8867-4", "body temperature": "8310-5",
                        "respiratory rate": "9279-1", "oxygen saturation": "2708-6",
                        "body weight": "29463-7", "weight": "29463-7", "body height": "8302-2",
                        "body mass index": "39156-5", "bmi": "39156-5",
                        "fetal heart rate": "55283-6", "fundal height": "11881-0",
                        "hemoglobin": "718-7", "hgb": "718-7", "platelets": "777-3", "plts": "777-3",
                        "inr": "6301-6", "glucose": "2345-7", "creatinine": "2160-0",
                        "potassium": "2823-3", "sodium": "2951-2",
                        "protein in urine": "2514-8", "urine protein": "2514-8",
                        "white blood cell count": "6690-2", "wbc": "6690-2",
                    }

                    def resolve_loinc(name):
                        n = name.lower()
                        code = name_to_loinc.get(n)
                        if not code:
                            for key, c in name_to_loinc.items():
                                if key in n or n in key:
                                    code = c
                                    break
                        if not code:
                            for desc, c in loinc_lookup.items():
                                if n in desc or desc in n:
                                    code = c
                                    break
                        return code or ""

                    def resolve_rxnorm(drug):
                        d = drug.lower()
                        code = rxnorm_lookup.get(d)
                        if not code:
                            for name, c in rxnorm_lookup.items():
                                if name in d or d in name:
                                    code = c
                                    break
                        return code or ""

                    def resolve_icd10(dx):
                        d = dx.lower()
                        for name, c in icd10_lookup.items():
                            if d in name or name in d:
                                return c
                        return ""

                    def resolve_cpt(proc):
                        p = proc.lower()
                        code = cpt_lookup.get(p)
                        if not code:
                            for name, c in cpt_lookup.items():
                                if p in name or name in p:
                                    code = c
                                    break
                        return code or ""

                    col_note, col_review = st.columns([1, 2])
                    with col_note:
                        st.markdown("**Original Note**")
                        st.text_area("", value=note_row[0]['NOTE_TEXT'], height=500, disabled=True, key="doc_review_note_text")

                    with col_review:
                        st.markdown("**Extracted Items — Edit to correct**")

                        corrections_made = []

                        obs = extraction.get('observations', [])
                        if obs:
                            st.markdown("##### Observations (LOINC)")
                            for i, o in enumerate(obs):
                                with st.container():
                                    c1, c2, c3 = st.columns([2, 1, 1])
                                    orig_name = o.get('name', '')
                                    orig_val = str(o.get('value', ''))
                                    resolved_code = resolve_loinc(orig_name)
                                    with c1:
                                        new_name = st.text_input(f"Name", value=orig_name, key=f"obs_name_{i}")
                                    with c2:
                                        new_val = st.text_input(f"Value", value=orig_val, key=f"obs_val_{i}")
                                    with c3:
                                        new_code = st.text_input(f"LOINC", value=resolved_code, key=f"obs_code_{i}")
                                    if new_name != orig_name or new_code != resolved_code:
                                        corrections_made.append({
                                            "resource_type": "Observation",
                                            "field_path": f"observations[{i}]",
                                            "original_value": orig_name,
                                            "original_code": resolved_code,
                                            "corrected_value": new_name if new_name != orig_name else orig_name,
                                            "corrected_code": new_code if new_code != resolved_code else resolved_code,
                                        })

                        meds = extraction.get('medications', [])
                        if meds:
                            st.markdown("##### Medications (RxNorm)")
                            for i, m in enumerate(meds):
                                orig_drug = m.get('drug', '') if isinstance(m, dict) else str(m)
                                resolved_code = resolve_rxnorm(orig_drug)
                                c1, c2 = st.columns([2, 1])
                                with c1:
                                    new_drug = st.text_input(f"Drug", value=orig_drug, key=f"med_name_{i}")
                                with c2:
                                    new_code = st.text_input(f"RxNorm", value=resolved_code, key=f"med_code_{i}")
                                if new_drug != orig_drug or new_code != resolved_code:
                                    corrections_made.append({
                                        "resource_type": "MedicationStatement",
                                        "field_path": f"medications[{i}]",
                                        "original_value": orig_drug,
                                        "original_code": resolved_code,
                                        "corrected_value": new_drug if new_drug != orig_drug else orig_drug,
                                        "corrected_code": new_code if new_code != resolved_code else resolved_code,
                                    })

                        dx = extraction.get('diagnoses', [])
                        if dx:
                            st.markdown("##### Diagnoses (ICD-10 / SNOMED)")
                            for i, d in enumerate(dx):
                                orig_dx = d.get('name', '') if isinstance(d, dict) else str(d)
                                orig_icd = d.get('icd10', '') if isinstance(d, dict) else ''
                                resolved_code = orig_icd or resolve_icd10(orig_dx)
                                c1, c2 = st.columns([2, 1])
                                with c1:
                                    new_dx = st.text_input(f"Diagnosis", value=orig_dx, key=f"dx_name_{i}")
                                with c2:
                                    new_code = st.text_input(f"ICD-10", value=resolved_code, key=f"dx_code_{i}")
                                if new_dx != orig_dx or new_code != resolved_code:
                                    corrections_made.append({
                                        "resource_type": "Condition",
                                        "field_path": f"diagnoses[{i}]",
                                        "original_value": orig_dx,
                                        "original_code": resolved_code,
                                        "corrected_value": new_dx if new_dx != orig_dx else orig_dx,
                                        "corrected_code": new_code if new_code != resolved_code else resolved_code,
                                    })

                        procs = extraction.get('procedures', [])
                        if procs:
                            st.markdown("##### Procedures (CPT)")
                            for i, p in enumerate(procs):
                                orig_proc = p if isinstance(p, str) else str(p)
                                resolved_code = resolve_cpt(orig_proc)
                                c1, c2 = st.columns([2, 1])
                                with c1:
                                    new_proc = st.text_input(f"Procedure", value=orig_proc, key=f"proc_name_{i}")
                                with c2:
                                    new_code = st.text_input(f"CPT", value=resolved_code, key=f"proc_code_{i}")
                                if new_proc != orig_proc or new_code != resolved_code:
                                    corrections_made.append({
                                        "resource_type": "Procedure",
                                        "field_path": f"procedures[{i}]",
                                        "original_value": orig_proc,
                                        "original_code": resolved_code,
                                        "corrected_value": new_proc if new_proc != orig_proc else orig_proc,
                                        "corrected_code": new_code if new_code != resolved_code else resolved_code,
                                    })

                        st.markdown("---")
                        if corrections_made:
                            st.info(f"{len(corrections_made)} correction(s) pending")

                        reason = st.text_input("Correction reason (optional)", key="doc_review_reason", placeholder="e.g. Wrong drug mapped, AI hallucinated diagnosis")

                        if st.button("Save Corrections", key="doc_review_save", type="primary"):
                            if corrections_made:
                                for c in corrections_made:
                                    reason_escaped = reason.replace("'", "''") if reason else ""
                                    session.sql(f"""
                                        INSERT INTO {DOC_DB}.{DOC_SCHEMA}.EXTRACTION_CORRECTIONS
                                        (note_id, resource_type, field_path, original_value, original_code, corrected_value, corrected_code, correction_reason)
                                        VALUES ({note_id}, '{c["resource_type"]}', '{c["field_path"]}',
                                                '{c["original_value"].replace("'", "''")}', '{c["original_code"]}',
                                                '{c["corrected_value"].replace("'", "''")}', '{c["corrected_code"]}',
                                                '{reason_escaped}')
                                    """).collect()
                                session.sql(f"""
                                    UPDATE {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
                                    SET extraction_status = 'REVIEWED', reviewed_at = CURRENT_TIMESTAMP()
                                    WHERE note_id = {note_id}
                                """).collect()
                                st.success(f"Saved {len(corrections_made)} correction(s) with before/after pairs!")
                            else:
                                session.sql(f"""
                                    UPDATE {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
                                    SET extraction_status = 'REVIEWED', reviewed_at = CURRENT_TIMESTAMP()
                                    WHERE note_id = {note_id}
                                """).collect()
                                st.success("Marked as reviewed (no corrections needed)")

                        existing_corrections = session.sql(f"""
                            SELECT resource_type, field_path, original_value, original_code, corrected_value, corrected_code, correction_reason, corrected_at
                            FROM {DOC_DB}.{DOC_SCHEMA}.EXTRACTION_CORRECTIONS
                            WHERE note_id = {note_id}
                            ORDER BY corrected_at DESC
                        """).to_pandas()
                        if not existing_corrections.empty:
                            with st.expander(f"Previous corrections ({len(existing_corrections)})"):
                                st.dataframe(existing_corrections, use_container_width=True)

                elif note_row:
                    st.info("This note hasn't been extracted yet. Go to Upload & Extract → Select existing → Extract first.")
            else:
                st.info("No extracted notes to review. Run extraction on some notes first!")
        except Exception as e:
            st.warning(f"Could not load notes: {str(e)[:200]}")

    with doc_tab_history:
        st.markdown("#### Extraction History & Quality Metrics")
        try:
            stats = session.sql(f"""
                SELECT
                    COUNT(*) as total_notes,
                    COUNT(DISTINCT department) as departments,
                    COUNT(DISTINCT note_type) as note_types,
                    SUM(CASE WHEN extraction_status = 'EXTRACTED' THEN 1 ELSE 0 END) as extracted,
                    SUM(CASE WHEN extraction_status = 'REVIEWED' THEN 1 ELSE 0 END) as reviewed,
                    SUM(CASE WHEN extraction_status = 'PENDING' OR extraction_status IS NULL THEN 1 ELSE 0 END) as pending
                FROM {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
            """).collect()
            if stats:
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Total Notes", stats[0]['TOTAL_NOTES'])
                c2.metric("Pending", stats[0]['PENDING'])
                c3.metric("Extracted", stats[0]['EXTRACTED'])
                c4.metric("Reviewed", stats[0]['REVIEWED'])
                c5.metric("Departments", stats[0]['DEPARTMENTS'])
                c6.metric("Note Types", stats[0]['NOTE_TYPES'])

            st.markdown("---")
            st.markdown("##### Correction Quality Scores")

            correction_stats = session.sql(f"""
                SELECT
                    COUNT(*) as total_corrections,
                    COUNT(DISTINCT note_id) as notes_with_corrections,
                    COUNT(CASE WHEN resource_type = 'Observation' THEN 1 END) as obs_corrections,
                    COUNT(CASE WHEN resource_type = 'MedicationStatement' THEN 1 END) as med_corrections,
                    COUNT(CASE WHEN resource_type = 'Condition' THEN 1 END) as dx_corrections,
                    COUNT(CASE WHEN resource_type = 'Procedure' THEN 1 END) as proc_corrections
                FROM {DOC_DB}.{DOC_SCHEMA}.EXTRACTION_CORRECTIONS
            """).collect()

            if correction_stats and correction_stats[0]['TOTAL_CORRECTIONS'] > 0:
                cs = correction_stats[0]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Corrections", cs['TOTAL_CORRECTIONS'])
                c2.metric("Observation Fixes", cs['OBS_CORRECTIONS'])
                c3.metric("Medication Fixes", cs['MED_CORRECTIONS'])
                c4.metric("Diagnosis Fixes", cs['DX_CORRECTIONS'])
                c5.metric("Procedure Fixes", cs['PROC_CORRECTIONS'])

                reviewed_count = stats[0]['REVIEWED'] if stats else 0
                extracted_count = stats[0]['EXTRACTED'] if stats else 0
                total_processed = reviewed_count + extracted_count
                if total_processed > 0:
                    notes_needing_correction = cs['NOTES_WITH_CORRECTIONS']
                    accuracy_pct = int((1 - notes_needing_correction / total_processed) * 100) if total_processed > 0 else 0
                    st.metric("AI Extraction Accuracy (notes needing zero corrections)", f"{accuracy_pct}%")

                st.markdown("##### Recent Corrections")
                recent = session.sql(f"""
                    SELECT c.note_id, n.note_type, c.resource_type, c.field_path,
                           c.original_value, c.original_code, c.corrected_value, c.corrected_code,
                           c.correction_reason, c.corrected_at
                    FROM {DOC_DB}.{DOC_SCHEMA}.EXTRACTION_CORRECTIONS c
                    JOIN {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES n ON c.note_id = n.note_id
                    ORDER BY c.corrected_at DESC
                    LIMIT 20
                """).to_pandas()
                if not recent.empty:
                    st.dataframe(recent, use_container_width=True)
            else:
                st.info("No corrections recorded yet. Review some extracted notes to build quality data.")

            st.markdown("---")
            st.markdown("##### Notes by Department & Type")
            by_dept = session.sql(f"""
                SELECT department, note_type, extraction_status, COUNT(*) as cnt
                FROM {DOC_DB}.{DOC_SCHEMA}.CLINICAL_NOTES
                GROUP BY department, note_type, extraction_status
                ORDER BY cnt DESC
            """).to_pandas()
            if not by_dept.empty:
                st.dataframe(by_dept, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load history: {str(e)[:200]}")

with tab_config:

    with st.expander("First-time setup — grant access to your data", expanded=False):
        st.markdown("""
This app runs with its own privileges. To allow it to read your HL7v2/FHIR source data
and write OMOP/Tuva output, you need to grant it access to the relevant database(s).

**Run these SQL statements** in a worksheet (replace the placeholders):
""")
        setup_db = st.text_input("Your database name", value="MY_DATABASE", key="setup_db_name")
        setup_schema = st.text_input("Schema containing HL7v2/FHIR data", value="FHIR_STAGING", key="setup_schema_name")
        setup_table = st.text_input("Table containing HL7v2/FHIR data", value="RAW_BUNDLES", key="setup_table_name")
        app_name = "TUVA_FHIR_TO_OMOP_APP"

        grant_sql = f"""-- Grant read access to your HL7v2/FHIR source data
GRANT USAGE ON DATABASE {setup_db} TO APPLICATION {app_name};
GRANT USAGE ON SCHEMA {setup_db}.{setup_schema} TO APPLICATION {app_name};
GRANT SELECT ON TABLE {setup_db}.{setup_schema}.{setup_table} TO APPLICATION {app_name};

-- Grant write access for output (choose one):
-- Option A: Let the app create schemas in your database
GRANT CREATE SCHEMA ON DATABASE {setup_db} TO APPLICATION {app_name};
-- Option B: Grant access to an existing output schema
-- GRANT USAGE ON SCHEMA {setup_db}.OMOP_STAGING TO APPLICATION {app_name};
-- GRANT CREATE TABLE ON SCHEMA {setup_db}.OMOP_STAGING TO APPLICATION {app_name};

-- Bind the warehouse reference
CALL {app_name}.CORE.REGISTER_REFERENCE('CONSUMER_WAREHOUSE', 'SET', 'COMPUTE_WH');

-- Bind the source table reference
CALL {app_name}.CORE.REGISTER_REFERENCE('FHIR_SOURCE_DATABASE', 'SET', '{setup_db}.{setup_schema}.{setup_table}');"""  

        st.code(grant_sql, language="sql")

        st.info(
            "After running these grants, refresh this page. "
            "Your database will then appear in the dropdowns below."
        )

    st.subheader("HL7v2 / FHIR source")

    databases = list_databases()

    json_column = 'BUNDLE_DATA'
    bundle_id_col = 'BUNDLE_ID'
    detected_format = 'unknown'

    src_col1, src_col2, src_col3 = st.columns(3)
    with src_col1:
        source_db = st.selectbox("Database", databases, index=None,
                                  placeholder="Select a database…", key="cfg_src_db")
    with src_col2:
        src_schemas = list_schemas(source_db) if source_db else []
        source_schema = st.selectbox("Schema", src_schemas, index=None,
                                      placeholder="Select a schema…", key="cfg_src_schema")
    with src_col3:
        src_tables = list_tables(source_db, source_schema) if source_db and source_schema else []
        source_table = st.selectbox("Table", src_tables, index=None,
                                     placeholder="Select a table…", key="cfg_src_table")

    if source_db and source_schema and source_table:
        cols = list_columns(source_db, source_schema, source_table)
        col_types = list_columns_with_types(source_db, source_schema, source_table)

        variant_cols = [name for name, typ in col_types if 'VARIANT' in typ]
        varchar_cols = [name for name, typ in col_types if 'VARCHAR' in typ]
        data_cols = variant_cols + varchar_cols if variant_cols or varchar_cols else cols

        data_col_default = 0
        for i, c in enumerate(data_cols):
            if c in ('BUNDLE_DATA', 'RAW_MESSAGE', 'RAW_DATA', 'MESSAGE', 'DATA'):
                data_col_default = i
                break

        jc1, jc2 = st.columns(2)
        with jc1:
            json_column = st.selectbox("Data Column",
                                        data_cols,
                                        index=data_col_default,
                                        key="cfg_json_col")

        detected_format = detect_source_format(source_db, source_schema, source_table, json_column)
        format_labels = {'json': 'FHIR JSON', 'xml': 'FHIR XML', 'hl7v2': 'HL7v2', 'unknown': 'Unknown'}

        with jc1:
            st.caption(f"Detected: **{format_labels.get(detected_format, 'Unknown')}** — auto-detected from first row")

        with jc2:
            id_options = [c for c in cols if c != json_column]
            id_default = 0
            id_pref = ['BUNDLE_ID', 'MESSAGE_ID', 'ID', 'RECORD_ID', 'ROW_ID']
            for pref in id_pref:
                if pref in id_options:
                    id_default = id_options.index(pref)
                    break
            else:
                for i, c in enumerate(id_options):
                    if 'ID' in c.upper():
                        id_default = i
                        break
            id_label = "Message ID Column" if detected_format == 'hl7v2' else "Bundle ID Column"
            bundle_id_col = st.selectbox(id_label,
                                          id_options,
                                          index=id_default,
                                          key="cfg_bid_col")

        with st.expander("Preview source data", expanded=False):
            preview_df = get_table_preview(source_db, source_schema, source_table)
            if len(preview_df) > 0:
                row_count = session.sql(
                    f"SELECT COUNT(*) AS cnt FROM {source_db}.{source_schema}.{source_table}"
                ).collect()[0]['CNT']
                st.caption(f"{row_count:,} total rows")
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
            else:
                st.info("Table is empty or not accessible.")

    st.markdown("---")
    st.subheader("Output format")

    format_options = ["OMOP CDM v5.4", "Tuva Input Layer"]
    output_format = st.radio("Target data model", format_options, horizontal=True, key="cfg_output_format")
    output_format_key = "OMOP" if output_format == "OMOP CDM v5.4" else "TUVA"

    tuva_data_source = ""
    if output_format_key == "TUVA":
        tuva_data_source = st.text_input(
            "Data source label",
            value="fhir",
            help="Tuva requires a data_source tag on every row. Use a short label like 'fhir', 'claims', 'ehr'.",
            key="cfg_data_source",
        )
        st.caption("This label will be stamped on every Tuva output table for lineage tracking.")

    st.markdown("---")
    default_schema = "TUVA_INPUT" if output_format_key == "TUVA" else "OMOP_STAGING"
    st.subheader("Output destination")

    out_col1, out_col2 = st.columns(2)
    with out_col1:
        out_db = st.selectbox("Output Database", databases,
                               index=databases.index(source_db) if source_db and source_db in databases else 0,
                               key="cfg_out_db")
    with out_col2:
        out_schemas = list_schemas(out_db) if out_db else []
        new_schema_label = "➕ Create new schema…"
        schema_options = out_schemas + [new_schema_label]

        default_out_idx = None
        for i, s in enumerate(schema_options):
            if s == default_schema:
                default_out_idx = i
                break
        if default_out_idx is None:
            default_out_idx = schema_options.index(new_schema_label)

        out_schema_choice = st.selectbox("Output Schema", schema_options,
                                          index=default_out_idx,
                                          key="cfg_out_schema")

    if out_schema_choice == new_schema_label:
        new_schema_name = st.text_input("New schema name", value=default_schema,
                                         key="cfg_new_schema")
        output_schema = new_schema_name.upper().strip()
    else:
        output_schema = out_schema_choice

    if output_schema.upper() in PROTECTED_SCHEMAS:
        st.warning(
            f"⚠️ **{output_schema}** contains existing production data. "
            "Writing here will **replace all OMOP tables**."
        )
        try:
            prod_counts = []
            for tbl in ['person', 'condition_occurrence', 'measurement', 'visit_occurrence',
                        'drug_exposure', 'procedure_occurrence', 'observation', 'death',
                        'device_exposure', 'location', 'care_site', 'provider',
                        'payer_plan_period', 'cost', 'fact_relationship',
                        'observation_period', 'cdm_source']:
                try:
                    cnt = session.sql(f"SELECT COUNT(*) AS c FROM {out_db}.{output_schema}.{tbl}").collect()[0]['C']
                    if cnt > 0:
                        prod_counts.append(f"**{tbl}**: {cnt:,} rows")
                except:
                    pass
            if prod_counts:
                st.error("Existing data that will be **overwritten**:\n\n" + "  •  ".join(prod_counts))
        except:
            pass
        confirm_prod = st.checkbox(
            f"I understand — overwrite all tables in {output_schema}", value=False, key="cfg_confirm_prod"
        )
        if not confirm_prod:
            st.stop()

    st.markdown("---")

    if source_db and source_schema and source_table and output_schema:
        fq_table = f"{source_db}.{source_schema}.{source_table}"

        st.subheader("Review")
        rev1, rev2 = st.columns(2)
        with rev1:
            st.markdown(f"""
            **Source**
            - `{fq_table}`
            - Data column: `{json_column if source_table else '—'}` *({format_labels.get(detected_format, 'auto-detect')})*
            - {'Message' if detected_format == 'hl7v2' else 'Bundle'} ID: `{bundle_id_col if source_table else '—'}`
            """)
        with rev2:
            if output_format_key == "TUVA":
                tuva_tables = "patient, encounter, condition, lab_result, medication, observation, procedure, immunization, location, practitioner, medical_claim, eligibility, appointment"
                st.markdown(f"""
                **Destination** *(Tuva Input Layer)*
                - `{out_db}.{output_schema}`
                - Data source: `{tuva_data_source}`
                - Tables: {tuva_tables}
                """)
            else:
                st.markdown(f"""
                **Destination** *(OMOP CDM v5.4)*
                - `{out_db}.{output_schema}`
                - Tables: person, observation_period, condition_occurrence, measurement, visit_occurrence, drug_exposure, procedure_occurrence, device_exposure, observation, death, location, care_site, provider, payer_plan_period, cost, fact_relationship, cdm_source
                """)

        if st.button("Save configuration", type="primary", use_container_width=True):
            try:
                session.sql(f"""
                    MERGE INTO app_state.configuration t
                    USING (
                        SELECT 'source_table' AS key, '{fq_table}' AS value
                        UNION ALL SELECT 'json_column', '{json_column}'
                        UNION ALL SELECT 'bundle_id_column', '{bundle_id_col}'
                        UNION ALL SELECT 'output_schema', '{output_schema}'
                        UNION ALL SELECT 'output_database', '{out_db}'
                        UNION ALL SELECT 'output_format', '{output_format_key}'
                        UNION ALL SELECT 'data_source', '{tuva_data_source}'
                    ) s ON t.key = s.key
                    WHEN MATCHED THEN UPDATE SET value = s.value, updated_at = CURRENT_TIMESTAMP()
                    WHEN NOT MATCHED THEN INSERT (key, value) VALUES (s.key, s.value)
                """).collect()
                st.success("✅ Configuration saved!")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Error saving: {e}")
    else:
        st.info("Select a source table and output schema above to continue.")

with tab_run:
    config = load_config()
    run_format = config.get('output_format', 'OMOP')
    format_label = "Tuva Input Layer" if run_format == "TUVA" else "OMOP CDM v5.4"
    st.subheader(f"Run HL7v2/FHIR → {format_label}")

    if config.get('source_table'):
        out_schema_display = config.get('output_schema', 'OMOP_STAGING')
        out_db_display = config.get('output_database', '')
        fq_out = f"{out_db_display}.{out_schema_display}" if out_db_display else out_schema_display
        run_data_source = config.get('data_source', 'fhir')

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(f"**Source:** `{config['source_table']}`")
        with c2:
            st.markdown(f"**→ Output:** `{fq_out}` *({format_label})*")
        with c3:
            st.markdown(f"**Data col:** `{config.get('json_column', 'BUNDLE_DATA')}`")

        if out_schema_display.upper() in PROTECTED_SCHEMAS:
            st.error(f"⚠️ Output targets **{out_schema_display}** — production schema. Change in Configure tab if unintended.")

        st.markdown("---")

        OMOP_PIPELINE = [
            ("Parsing bundles", "core.parse_fhir_bundles", "parse"),
            ("Mapping persons", "core.map_persons", "map"),
            ("Mapping conditions", "core.map_conditions", "map"),
            ("Mapping measurements", "core.map_measurements", "map"),
            ("Mapping visits", "core.map_visits", "map"),
            ("Mapping drug exposures", "core.map_drug_exposures", "map"),
            ("Mapping procedures", "core.map_procedures", "map"),
            ("Mapping observations", "core.map_observations_qual", "map"),
            ("Mapping death records", "core.map_death", "map"),
            ("Mapping immunizations", "core.map_immunizations", "map"),
            ("Mapping med admin", "core.map_med_administrations", "map"),
            ("Mapping allergies", "core.map_allergies", "map"),
            ("Mapping devices", "core.map_devices", "map"),
            ("Mapping diag reports", "core.map_diagnostic_reports", "map"),
            ("Mapping imaging", "core.map_imaging_studies", "map"),
            ("Mapping care plans", "core.map_care_plans", "map"),
            ("Mapping locations", "core.map_locations", "map"),
            ("Mapping organizations", "core.map_organizations", "map"),
            ("Mapping practitioners", "core.map_practitioners", "map"),
            ("Mapping claims/EOB", "core.map_claims", "map"),
            ("Mapping care teams", "core.map_care_teams", "map"),
            ("Building obs periods", "core.build_observation_periods", "map"),
            ("Building CDM source", "core.build_cdm_source", "map"),
        ]

        TUVA_PIPELINE = [
            ("Parsing bundles", "core.parse_fhir_bundles", "parse"),
            ("Mapping patients", "core.map_tuva_patient", "tuva"),
            ("Mapping encounters", "core.map_tuva_encounter", "tuva"),
            ("Mapping conditions", "core.map_tuva_condition", "tuva"),
            ("Mapping lab results", "core.map_tuva_lab_result", "tuva"),
            ("Mapping observations", "core.map_tuva_observation", "tuva"),
            ("Mapping medications", "core.map_tuva_medication", "tuva"),
            ("Mapping immunizations", "core.map_tuva_immunization", "tuva"),
            ("Mapping procedures", "core.map_tuva_procedure", "tuva"),
            ("Mapping locations", "core.map_tuva_location", "tuva"),
            ("Mapping practitioners", "core.map_tuva_practitioner", "tuva"),
            ("Mapping medical claims", "core.map_tuva_medical_claim", "tuva"),
            ("Mapping eligibility", "core.map_tuva_eligibility", "tuva"),
            ("Mapping appointments", "core.map_tuva_appointment", "tuva"),
        ]

        OMOP_SUMMARY_TABLES = [
            'person', 'observation_period', 'condition_occurrence', 'measurement',
            'visit_occurrence', 'drug_exposure', 'procedure_occurrence',
            'device_exposure', 'observation', 'death',
            'location', 'care_site', 'provider',
            'payer_plan_period', 'cost', 'fact_relationship', 'cdm_source',
        ]

        TUVA_SUMMARY_TABLES = [
            'patient', 'encounter', 'condition', 'lab_result',
            'observation', 'medication', 'immunization', 'procedure',
            'location', 'practitioner', 'medical_claim',
            'eligibility', 'appointment',
        ]

        pipeline_steps = TUVA_PIPELINE if run_format == "TUVA" else OMOP_PIPELINE
        summary_tables = TUVA_SUMMARY_TABLES if run_format == "TUVA" else OMOP_SUMMARY_TABLES

        run_col1, run_col2 = st.columns([1, 2])
        with run_col1:
            run_btn = st.button("Run full pipeline", type="primary", use_container_width=True)
        with run_col2:
            step_names = " → ".join(s[0].split(" ", 1)[1].title() for s in pipeline_steps)
            st.caption(step_names)

        if run_btn:
            import time as _time

            run_id = session.sql("SELECT UUID_STRING()").collect()[0][0]
            try:
                session.sql(f"""
                    INSERT INTO app_state.run_history (run_id, status)
                    VALUES ('{run_id}', 'RUNNING')
                """).collect()
            except:
                pass

            total_steps = len(pipeline_steps)
            progress = st.progress(0, text="Starting pipeline…")
            log_expander = st.expander("Pipeline log", expanded=True)
            with log_expander:
                log_placeholder = st.empty()
            log_lines = []
            errors = []
            t_start = _time.time()

            for step_idx, (label, proc, kind) in enumerate(pipeline_steps):
                pct = int((step_idx / total_steps) * 100)
                progress.progress(pct, text=f"{label}… ({step_idx + 1}/{total_steps})")

                step_t = _time.time()
                try:
                    if kind == 'parse':
                        result = session.call(
                            proc,
                            config['source_table'],
                            config.get('json_column', 'BUNDLE_DATA'),
                            config.get('bundle_id_column', 'BUNDLE_ID')
                        )
                    elif kind == 'tuva':
                        result = session.call(proc, config.get('output_schema', 'TUVA_INPUT'), run_data_source)
                    else:
                        result = session.call(proc, config.get('output_schema', 'OMOP_STAGING'))
                    elapsed = _time.time() - step_t
                    log_lines.append(f"✓ ({step_idx + 1}/{total_steps})  {label} — {elapsed:.1f}s — {result}")
                except Exception as e:
                    elapsed = _time.time() - step_t
                    err_short = str(e).split('\n')[0][:120]
                    log_lines.append(f"✗ ({step_idx + 1}/{total_steps})  {label} — {elapsed:.1f}s — ERROR: {err_short}")
                    errors.append(f"{label}: {err_short}")

                log_placeholder.text('\n'.join(log_lines))

            total_elapsed = _time.time() - t_start
            progress.progress(100, text=f"Pipeline complete! ({total_elapsed:.0f}s)")

            if errors:
                st.warning(f"Completed with {len(errors)} error(s):\n\n" + "\n".join(f"- {e}" for e in errors))
            else:
                st.success(f"All {total_steps} steps completed successfully in {total_elapsed:.0f} seconds.")
                st.balloons()

            with st.expander("Output summary", expanded=True):
                summary_data = []
                for tbl in summary_tables:
                    try:
                        cnt = session.sql(
                            f"SELECT COUNT(*) AS c FROM {out_schema_display}.{tbl}"
                        ).collect()[0]['C']
                        summary_data.append({"Table": tbl, "Rows": cnt})
                    except:
                        summary_data.append({"Table": tbl, "Rows": "—"})
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            try:
                counts = {d['Table']: d['Rows'] for d in summary_data if isinstance(d.get('Rows'), (int, float))}
                status = 'COMPLETED_WITH_ERRORS' if errors else 'COMPLETED'
                if run_format == "TUVA":
                    session.sql(f"""
                        UPDATE app_state.run_history SET
                            status = '{status}',
                            completed_at = CURRENT_TIMESTAMP(),
                            persons_mapped = {counts.get('patient', 0)},
                            conditions_mapped = {counts.get('condition', 0)},
                            measurements_mapped = {counts.get('lab_result', 0)},
                            visits_mapped = {counts.get('encounter', 0)},
                            errors = {len(errors)}
                        WHERE run_id = '{run_id}'
                    """).collect()
                else:
                    session.sql(f"""
                        UPDATE app_state.run_history SET
                            status = '{status}',
                            completed_at = CURRENT_TIMESTAMP(),
                            fhir_bundles = {counts.get('person', 0)},
                            persons_mapped = {counts.get('person', 0)},
                            conditions_mapped = {counts.get('condition_occurrence', 0)},
                            measurements_mapped = {counts.get('measurement', 0)},
                            visits_mapped = {counts.get('visit_occurrence', 0)},
                            errors = {len(errors)}
                        WHERE run_id = '{run_id}'
                    """).collect()
            except:
                pass

        st.markdown("---")
        st.subheader("Individual mappers")
        st.caption("Run a single step of the pipeline for debugging or re-processing.")

        mappers = pipeline_steps[:]
        cols_per_row = 5
        rows_needed = (len(mappers) + cols_per_row - 1) // cols_per_row
        all_cols = []
        for _ in range(rows_needed - 1):
            all_cols += st.columns(cols_per_row)
        remaining = len(mappers) - (rows_needed - 1) * cols_per_row
        if remaining > 0:
            all_cols += st.columns(remaining)

        for i, (label, proc, kind) in enumerate(mappers):
            with all_cols[i]:
                if st.button(label.split(" ", 1)[1] if " " in label else label, use_container_width=True, key=f"mapper_{i}"):
                    with st.spinner(f"Running {label}…"):
                        try:
                            if kind == 'parse':
                                result = session.call(
                                    proc,
                                    config['source_table'],
                                    config.get('json_column', 'BUNDLE_DATA'),
                                    config.get('bundle_id_column', 'BUNDLE_ID')
                                )
                            elif kind == 'tuva':
                                result = session.call(proc, config.get('output_schema', 'TUVA_INPUT'), run_data_source)
                            else:
                                result = session.call(proc, config.get('output_schema', 'OMOP_STAGING'))
                            st.success(result)
                        except Exception as e:
                            st.error(str(e))
    else:
        st.info("Configure your HL7v2/FHIR source in the **Configure** tab first.")

with tab_history:
    st.subheader("Run history")
    try:
        history = session.sql("""
            SELECT run_id, started_at, completed_at, status,
                   fhir_bundles, persons_mapped, conditions_mapped,
                   measurements_mapped, visits_mapped, errors
            FROM app_state.run_history
            ORDER BY started_at DESC
            LIMIT 50
        """).to_pandas()

        if len(history) > 0:
            col1, col2, col3, col4 = st.columns(4)
            latest = history.iloc[0]
            col1.metric("Total Runs", len(history))
            col2.metric("Last Status", latest['STATUS'])
            col3.metric("Last Persons", f"{latest['PERSONS_MAPPED']:,}")
            col4.metric("Last Conditions", f"{latest['CONDITIONS_MAPPED']:,}")
            st.dataframe(history, use_container_width=True, hide_index=True)
        else:
            st.info("No transformation runs yet. Go to the **Run** tab to start.")
    except:
        st.info("Run history will appear here after your first transformation.")


VOCAB_MAP = {
    "LOINC": "terminology.loinc_to_omop",
    "SNOMED": "terminology.snomed_to_omop",
    "RxNorm": "terminology.rxnorm_to_omop",
    "ICD-10-CM": "terminology.icd10cm_to_omop",
    "CPT": "terminology.cpt_to_omop",
    "HCPCS": "terminology.hcpcs_to_omop",
    "Demographics": "terminology.demographic_to_omop",
}

with tab_vocab:
    st.subheader("Vocabulary browser")

    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        selected_vocab = st.selectbox("Vocabulary", list(VOCAB_MAP.keys()))
    with v_col2:
        search_term = st.text_input("Search by code or description", key="vocab_search", placeholder="Type to filter…")

    if selected_vocab:
        vocab_table = VOCAB_MAP[selected_vocab]
        try:
            count_row = session.sql(f"SELECT COUNT(*) AS cnt FROM {vocab_table}").collect()
            total_count = count_row[0]['CNT']
            st.caption(f"**{selected_vocab}** — {total_count:,} total records")

            page_size = 50
            if f"vocab_page_{selected_vocab}" not in st.session_state:
                st.session_state[f"vocab_page_{selected_vocab}"] = 0
            page = st.session_state[f"vocab_page_{selected_vocab}"]

            cols = session.sql(f"SELECT * FROM {vocab_table} LIMIT 0").to_pandas().columns.tolist()

            where = ""
            if search_term:
                like = search_term.replace("'", "''")
                clauses = [f"CAST({c} AS VARCHAR) ILIKE '%{like}%'" for c in cols]
                where = "WHERE " + " OR ".join(clauses)

            filtered_count = session.sql(
                f"SELECT COUNT(*) AS cnt FROM {vocab_table} {where}"
            ).collect()[0]['CNT']
            total_pages = max(1, (filtered_count + page_size - 1) // page_size)
            page = min(page, total_pages - 1)

            offset = page * page_size
            df_vocab = session.sql(
                f"SELECT * FROM {vocab_table} {where} LIMIT {page_size} OFFSET {offset}"
            ).to_pandas()
            st.dataframe(df_vocab, use_container_width=True, hide_index=True)

            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("Previous", disabled=(page == 0), key="vocab_prev"):
                    st.session_state[f"vocab_page_{selected_vocab}"] = page - 1
                    st.rerun()
            with nav_col2:
                st.caption(f"Page {page + 1} of {total_pages} ({filtered_count:,} matching)")
            with nav_col3:
                if st.button("Next", disabled=(page >= total_pages - 1), key="vocab_next"):
                    st.session_state[f"vocab_page_{selected_vocab}"] = page + 1
                    st.rerun()

        except Exception as e:
            st.info(f"Vocabulary table `{vocab_table}` not available yet. Deploy the app to populate terminology seeds.")


COVERAGE_DOMAINS = {
    "Person": {"table": "person", "concept_col": "gender_concept_id"},
    "Condition": {"table": "condition_occurrence", "concept_col": "condition_concept_id"},
    "Measurement": {"table": "measurement", "concept_col": "measurement_concept_id"},
    "Visit": {"table": "visit_occurrence", "concept_col": "visit_concept_id"},
    "Drug Exposure": {"table": "drug_exposure", "concept_col": "drug_concept_id"},
    "Procedure": {"table": "procedure_occurrence", "concept_col": "procedure_concept_id"},
    "Observation": {"table": "observation", "concept_col": "observation_concept_id"},
    "Device": {"table": "device_exposure", "concept_col": "device_concept_id"},
}

with tab_coverage:
    config = load_config()
    cov_format = config.get('output_format', 'OMOP')
    cov_schema = config.get('output_schema', 'OMOP_STAGING')

    if cov_format == "TUVA":
        st.subheader("Tuva Input Layer — table summary")
        st.caption("Tuva uses source codes directly — no concept_id coverage analysis needed.")

        tuva_tables_cov = [
            ('patient', 'source_code'), ('encounter', 'source_code'), ('condition', 'source_code'),
            ('lab_result', 'source_code'), ('observation', 'source_code'),
            ('medication', 'source_code'), ('immunization', 'source_code'),
            ('procedure', 'source_code'), ('location', 'name'),
            ('practitioner', 'npi'), ('medical_claim', 'hcpcs_code'),
            ('eligibility', 'payer'), ('appointment', 'source_code'),
        ]
        tuva_cov_data = []
        for tbl, code_col in tuva_tables_cov:
            try:
                row = session.sql(f"""
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN {code_col} IS NOT NULL AND {code_col} != '' THEN 1 ELSE 0 END) AS with_code
                    FROM {cov_schema}.{tbl}
                """).collect()[0]
                total = row['TOTAL'] or 0
                coded = row['WITH_CODE'] or 0
                pct = round(coded / total * 100, 1) if total > 0 else 0.0
                tuva_cov_data.append({"Table": tbl, "Total Rows": total, "With Code": coded, "Code %": pct})
            except:
                pass

        if tuva_cov_data:
            st.dataframe(pd.DataFrame(tuva_cov_data), use_container_width=True, hide_index=True)
        else:
            st.info("No Tuva tables found. Run the transformation first.")
    else:
        st.subheader("Coverage report")

        coverage_data = []
        for domain, info in COVERAGE_DOMAINS.items():
            tbl = f"{cov_schema}.{info['table']}"
            col = info['concept_col']
            try:
                row = session.sql(f"""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN {col} > 0 THEN 1 ELSE 0 END) AS mapped,
                        SUM(CASE WHEN {col} = 0 OR {col} IS NULL THEN 1 ELSE 0 END) AS unmapped
                    FROM {tbl}
                """).collect()[0]
                total = row['TOTAL'] or 0
                mapped = row['MAPPED'] or 0
                unmapped = row['UNMAPPED'] or 0
                pct = round(mapped / total * 100, 1) if total > 0 else 0.0
                coverage_data.append({"Domain": domain, "Total": total, "Mapped": mapped,
                                      "Unmapped": unmapped, "Coverage %": pct})
            except:
                pass

        if coverage_data:
            total_all = sum(d['Total'] for d in coverage_data)
            mapped_all = sum(d['Mapped'] for d in coverage_data)
            unmapped_all = sum(d['Unmapped'] for d in coverage_data)
            overall_pct = round(mapped_all / total_all * 100, 1) if total_all > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Rows", f"{total_all:,}")
            m2.metric("Mapped", f"{mapped_all:,}")
            m3.metric("Unmapped", f"{unmapped_all:,}")
            m4.metric("Overall Coverage", f"{overall_pct}%")

            chart_df = pd.DataFrame(coverage_data).set_index("Domain")[["Mapped", "Unmapped"]]
            st.bar_chart(chart_df)

            st.dataframe(pd.DataFrame(coverage_data), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Top 20 Unmapped Source Codes by Domain")

            SOURCE_CODE_MAP = {
                "Condition": {"table": "condition_occurrence", "concept_col": "condition_concept_id",
                              "source_col": "condition_source_value"},
                "Measurement": {"table": "measurement", "concept_col": "measurement_concept_id",
                                "source_col": "measurement_source_value"},
                "Drug Exposure": {"table": "drug_exposure", "concept_col": "drug_concept_id",
                                  "source_col": "drug_source_value"},
                "Visit": {"table": "visit_occurrence", "concept_col": "visit_concept_id",
                          "source_col": "visit_source_value"},
                "Procedure": {"table": "procedure_occurrence", "concept_col": "procedure_concept_id",
                              "source_col": "procedure_source_value"},
                "Observation": {"table": "observation", "concept_col": "observation_concept_id",
                                "source_col": "observation_source_value"},
                "Device": {"table": "device_exposure", "concept_col": "device_concept_id",
                           "source_col": "device_source_value"},
            }

            for domain, info in SOURCE_CODE_MAP.items():
                tbl = f"{cov_schema}.{info['table']}"
                col = info['concept_col']
                src = info['source_col']
                try:
                    unmapped_df = session.sql(f"""
                        SELECT {src} AS source_code, COUNT(*) AS occurrences
                        FROM {tbl}
                        WHERE ({col} = 0 OR {col} IS NULL) AND {src} IS NOT NULL
                        GROUP BY {src}
                        ORDER BY occurrences DESC
                        LIMIT 20
                    """).to_pandas()
                    if len(unmapped_df) > 0:
                        with st.expander(f"{domain} — {len(unmapped_df)} unmapped codes"):
                            st.dataframe(unmapped_df, use_container_width=True, hide_index=True)
                except:
                    pass
        else:
            st.info("No OMOP tables found. Run the transformation first to see coverage metrics.")


with tab_quality:
    st.subheader("Data quality")

    def run_quality_validation():
        try:
            result = session.call('core.validate_fhir_quality')
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            return None

    if st.button("Re-run validation", key="rerun_quality"):
        st.session_state['quality_result'] = run_quality_validation()

    if 'quality_result' not in st.session_state:
        st.session_state['quality_result'] = run_quality_validation()

    qr = st.session_state['quality_result']

    if qr:
        if isinstance(qr, dict):
            resource_dist = qr.get('resource_distribution', {})
            if resource_dist:
                st.subheader("Resource Type Distribution")
                dist_df = pd.DataFrame(
                    [{"Resource Type": k, "Count": v} for k, v in resource_dist.items()]
                ).sort_values("Count", ascending=False)
                st.bar_chart(dist_df.set_index("Resource Type"))

            issues = qr.get('issues', qr.get('quality_issues', []))
            if issues and isinstance(issues, list):
                st.subheader("Quality Issues")
                issues_df = pd.DataFrame(issues)
                severity_colors = {
                    'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'info': '🔵'
                }
                if 'severity' in issues_df.columns:
                    issues_df['severity'] = issues_df['severity'].apply(
                        lambda s: f"{severity_colors.get(str(s).lower(), '⚪')} {s}"
                    )
                st.dataframe(issues_df, use_container_width=True, hide_index=True)
            elif not issues:
                st.success("No quality issues detected.")

            summary = qr.get('summary', {})
            if summary:
                st.subheader("Summary")
                s_cols = st.columns(len(summary))
                for i, (k, v) in enumerate(summary.items()):
                    s_cols[i].metric(k.replace('_', ' ').title(), f"{v:,}" if isinstance(v, (int, float)) else str(v))

            other_keys = [k for k in qr.keys() if k not in ('resource_distribution', 'issues', 'quality_issues', 'summary')]
            if other_keys:
                with st.expander("Full Validation Result"):
                    st.json(qr)
        else:
            st.json(qr)
    else:
        st.info("Validation not available. Ensure `core.validate_fhir_quality` is deployed and FHIR data has been parsed.")


with tab_explore:
    config = load_config()
    explore_format = config.get('output_format', 'OMOP')
    out_schema = config.get('output_schema', 'OMOP_STAGING')

    if explore_format == "TUVA":
        st.subheader("Explore Tuva Input Layer output")
        all_tables = [
            'patient', 'encounter', 'condition', 'lab_result',
            'observation', 'medication', 'immunization', 'procedure',
            'location', 'practitioner', 'medical_claim',
            'eligibility', 'appointment',
        ]
    else:
        st.subheader("Explore OMOP CDM output")
        all_tables = [
            'person', 'observation_period', 'condition_occurrence', 'measurement',
            'visit_occurrence', 'drug_exposure', 'procedure_occurrence',
            'device_exposure', 'observation', 'death',
            'location', 'care_site', 'provider',
            'payer_plan_period', 'cost', 'fact_relationship', 'cdm_source',
        ]

    available_tables = []
    for tbl in all_tables:
        try:
            session.sql(f"SELECT 1 FROM {out_schema}.{tbl} LIMIT 0").collect()
            available_tables.append(tbl)
        except:
            pass

    if not available_tables:
        st.info("No output tables found. Run the transformation first.")
    else:
        exp_col1, exp_col2 = st.columns([1, 2])
        with exp_col1:
            selected_table = st.selectbox("Table", available_tables, key="explore_table")
        with exp_col2:
            row_limit = st.slider("Rows to display", min_value=10, max_value=500, value=100, step=10)

        if selected_table:
            try:
                count = session.sql(
                    f"SELECT COUNT(*) AS cnt FROM {out_schema}.{selected_table}"
                ).collect()[0]['CNT']

                st.caption(f"**{selected_table}** — {count:,} total rows (showing up to {row_limit})")

                df = session.sql(
                    f"SELECT * FROM {out_schema}.{selected_table} LIMIT {row_limit}"
                ).to_pandas()
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.info(f"Table `{out_schema}.{selected_table}` not yet created. Run the transformation first.")


st.divider()
st.caption(
    "HL7v2/FHIR to OMOP/Tuva · "
    "Vocabulary: [Tuva Health](https://thetuvaproject.com/) (Apache 2.0) + "
    "[OHDSI Athena](https://athena.ohdsi.org/) · "
    "OMOP CDM v5.4 + Tuva Input Layer · "
    "HL7v2 + FHIR R4 (JSON & XML) · "
    "Built on [Snowflake](https://www.snowflake.com/) · "
    "Multi-cloud: AWS | Azure | GCP"
)
