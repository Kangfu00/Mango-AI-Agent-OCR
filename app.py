import json
import os
import re
import base64

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Mango AI Agent (Typhoon Vision Edition)", layout="wide")

# --- 1. ตั้งค่า API Key ---
GENAI_TYPHON_API_KEY = os.getenv("GENAI_TYPHON_API_KEY")
if not GENAI_TYPHON_API_KEY:
    st.error("GENAI_TYPHON_API_KEY not found in .env file")
    st.stop()

# --- 2. ฟังก์ชัน Step 1: ใช้ Typhoon-OCR สแกนรูปภาพ (Vision) ---
def extract_text_with_typhoon_ocr(file_bytes, file_name):
    """ส่งรูปภาพไปให้ Typhoon-OCR ด้วยวิธี Official API"""
    url = "https://api.opentyphoon.ai/v1/ocr"
    
    headers = {
        'Authorization': f'Bearer {GENAI_TYPHON_API_KEY}'
    }
    
    # ทริคของ Streamlit: ส่ง bytes เป็น Tuple (ชื่อไฟล์, ข้อมูลดิบ)
    files = {
        'file': (file_name, file_bytes) 
    }
    
    data = {
        'model': 'typhoon-ocr',
        'task_type': 'default',
        'max_tokens': '4000', 
        'temperature': '0.1',
        'top_p': '0.1',
        'repetition_penalty': '1.2'
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            extracted_texts = []
            
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        parsed_content = json.loads(content)
                        text = parsed_content.get('natural_text', content)
                    except json.JSONDecodeError:
                        text = content
                    extracted_texts.append(text)
                elif not page_result.get('success'):
                    return f"Error API: {page_result.get('error', 'Unknown error')}"

            return '\n'.join(extracted_texts)
        else:
            return f"Error HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Error Exception: {str(e)}"

# --- 3. ฟังก์ชัน Step 2: ใช้ Typhoon สกัด JSON (Reasoning) ---
def analyze_with_typhoon_llm(ocr_text):
    system_instruction = """
    คุณเป็น AI ผู้เชี่ยวชาญด้านบัญชีไทย หน้าที่ของคุณคือการสกัดข้อมูลจากเอกสารใบเสร็จหรือใบกำกับภาษีให้ออกมาเป็นโครงสร้าง JSON เท่านั้น
    
    กฎข้อบังคับ:
    1. ห้ามมีข้อความทักทาย ห้ามอธิบาย ห้ามมี Markdown (```json) คืนค่ามาเฉพาะปีกกา { ... } เท่านั้น
    2. หากไม่พบข้อมูลส่วนใด ให้ใส่ค่าว่าง "" หรือ 0.0
    
    รูปแบบ JSON ที่ต้องการอย่างเคร่งครัด:
    {
        "invoice_no": "เลขที่ใบกำกับภาษี",
        "date": "วันที่ (DD/MM/YYYY)",
        "seller_name": "ชื่อบริษัทผู้ขาย",
        "seller_tax_id": "เลขที่ผู้เสียภาษีผู้ขาย",
        "buyer_tax_id": "เลขที่ผู้เสียภาษีผู้ซื้อ",
        "items": [
            { "name": "ชื่อสินค้า", "qty": 1, "unit_price": 0.0, "total": 0.0 }
        ],
        "subtotal": ยอดรวมก่อนภาษี (float),
        "vat_amount": ภาษีมูลค่าเพิ่ม (float),
        "grand_total": ยอดรวมสุทธิ (float)
    }
    """

    user_message = f"""
    กรุณาสกัดข้อมูลจากข้อความดิบต่อไปนี้:
    
    <receipt_data>
    {ocr_text}
    </receipt_data>
    """

    try:
        response = requests.post(
            "https://api.opentyphoon.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GENAI_TYPHON_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "typhoon-v2.5-30b-a3b-instruct",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 4000,
                "temperature": 0.0,
                "top_p": 0.2,
                "repetition_penalty": 1.2,
            },
            timeout=30,
        )
        
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code} - {response.text}"}

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match:
            clean_json = match.group(1)
        else:
            clean_json = content.strip().replace("```json", "").replace("```", "")
            
        return json.loads(clean_json)
        
    except Exception as e:
        return {"error": str(e)}

# ข้อมูล ERP จำลอง
MOCK_ERP = {
    "my_company": {
        "tax_id": "0190901999999", 
        "name": "บจก. A จำกัด"
    },
    "vendors": {
        "001660313311": { 
            "name": "Gemeolo TH (SME MOVE)", 
            "po_amount": 61953.00,
            "status": "Approved"
        },
        "0123456587698": { 
            "name": "บริษัท ไอแท็กซ์ อินคอร์ปอเรชั่น จำกัด", 
            "po_amount": 1000.00,
            "status": "Pending"
        },
        "0105555555555": {
            "name": "บริษัท แมงโก้ คอนซัลแตนท์ จำกัด", 
            "po_amount": 1500.00,
            "status": "Approved"
        }
    }
}

def render_ui():
    st.title("🍊 Mango Intelligent Receipt Agent")

    uploaded_file = st.file_uploader("เลือกรูปใบเสร็จเพื่อทดสอบ (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    # ==========================================
    # 📝 ส่วนการประมวลผล (จะทำงานเมื่อมีการอัปโหลดรูปเท่านั้น)
    # ==========================================
    if uploaded_file is not None:
        
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read() 
        
        if len(file_bytes) == 0:
            st.error("ไฟล์ภาพว่างเปล่า กรุณาอัปโหลดใหม่")
            return

        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        
        if image is None:
            st.error("ไม่สามารถถอดรหัสภาพได้ กรุณาตรวจสอบว่าไฟล์เป็นภาพที่รองรับ")
            return

        # ด่านตรวจความคมชัดของภาพ
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        focus_measure = cv2.Laplacian(gray_image, cv2.CV_64F).var()

        st.write(f"🔍 **ค่าความคมชัดของภาพ:** {focus_measure:.2f}")

        if focus_measure < 50.0:
            st.error("⚠️ ภาพนี้เบลอหรือความละเอียดต่ำเกินไป! AI อาจอ่านข้อมูลผิดพลาดได้ กรุณาถ่ายภาพหรืออัปโหลดใหม่ให้ชัดเจนขึ้นครับ")
            return 
        elif focus_measure < 100.0:
            st.warning("⚠️ ภาพนี้ค่อนข้างเบลอ AI จะพยายามอ่านอย่างเต็มที่ แต่อาจมีข้อมูลตกหล่นได้")
        else:
            st.success("✅ ภาพคมชัดดีมาก พร้อมส่งให้ AI วิเคราะห์")

        col_img, col_data = st.columns([1, 1])

        with col_img:
            st.subheader("🖼️ ต้นฉบับเอกสาร")
            st.image(image, use_column_width=True)

        with col_data:
            st.subheader("📝 ตรวจสอบข้อมูลบัญชี")

            ai_data = None
            raw_text = ""

            with st.spinner("Step 1: กำลังสแกนภาพด้วย Typhoon-OCR..."):
                raw_text = extract_text_with_typhoon_ocr(file_bytes, uploaded_file.name)
                if "Error" in raw_text:
                    st.error(f"❌ Typhoon-OCR ขัดข้อง: {raw_text}")
                    return

            with st.spinner("Step 2: กำลังวิเคราะห์ข้อมูลบัญชีด้วย Typhoon 30B..."):
                ai_data = analyze_with_typhoon_llm(raw_text)
                if "error" not in ai_data:
                    st.success("✅ วิเคราะห์ข้อมูลสำเร็จเรียบร้อย!")
                else:
                    st.error(f"❌ Typhoon LLM ขัดข้อง: {ai_data['error']}")
                    return

            with st.expander("ดูข้อความดิบที่ได้จาก Typhoon-OCR"):
                st.text(raw_text)

            with st.form("verification_form"):
                st.markdown("#### 🔹 ข้อมูลเอกสาร (Header)")
                c1, c2 = st.columns(2)
                f_inv = c1.text_input("เลขที่ใบกำกับภาษี", value=ai_data.get("invoice_no", ""))
                f_date = c2.text_input("วันที่ (Date)", value=ai_data.get("date", ""))

                st.markdown("#### 🔹 ข้อมูลผู้ซื้อ/ผู้ขาย (Partners)")
                f_seller = st.text_input("บริษัทผู้ขาย (Seller)", value=ai_data.get("seller_name", ""))
                f_s_tax = st.text_input("เลขผู้เสียภาษีผู้ขาย", value=ai_data.get("seller_tax_id", ""))
                f_b_tax = st.text_input("เลขผู้เสียภาษีผู้ซื้อ (ตรวจสอบว่าบิลออกชื่อเราไหม)", value=ai_data.get("buyer_tax_id", ""))

                st.markdown("#### 🔹 รายการสินค้า (Line Items)")
                items_df = pd.DataFrame(ai_data.get("items", []))
                edited_items = st.data_editor(items_df, num_rows="dynamic", use_container_width=True)

                st.markdown("#### 🔹 สรุปยอดเงิน (Financial Summary)")
                cc1, cc2, cc3 = st.columns(3)
                f_sub = cc1.number_input("รวมเงิน (Subtotal)", value=float(ai_data.get("subtotal", 0.0)))
                f_vat = cc2.number_input("ภาษี (VAT 7%)", value=float(ai_data.get("vat_amount", 0.0)))
                f_total = cc3.number_input("ยอดสุทธิ (Total)", value=float(ai_data.get("grand_total", 0.0)))

                st.markdown("#### 🔍 ผลการตรวจสอบจากระบบ ERP")
            
                # 💡 [เพิ่ม 2 บรรทัดนี้] ล้างขีด (-) และช่องว่างทิ้งให้เหลือแต่ตัวเลขล้วนๆ
                clean_b_tax = f_b_tax.replace("-", "").replace(" ", "").strip()
                clean_s_tax = f_s_tax.replace("-", "").replace(" ", "").strip()
            
                # 1. เช็กว่าออกบิลในชื่อบริษัทเราถูกต้องไหม (ใช้ clean_b_tax)
                if clean_b_tax == MOCK_ERP["my_company"]["tax_id"]:
                    st.success("🏢 ชื่อผู้ซื้อ: ถูกต้อง (ออกบิลในนามบริษัทเรา)")
                elif not clean_b_tax:
                    st.warning("🏢 ชื่อผู้ซื้อ: ไม่ระบุเลขผู้เสียภาษีผู้ซื้อในบิล")
                else:
                    st.error("🏢 ชื่อผู้ซื้อ: ผิดพลาด! (อาจออกบิลผิดบริษัท)")

                # 2. เช็กข้อมูลผู้ขายและยอดเงิน (ใช้ clean_s_tax)
                erp_vendor = MOCK_ERP["vendors"].get(clean_s_tax) # 👈 เปลี่ยนมาใช้ตัวแปรที่คลีนแล้ว
                if erp_vendor:
                    st.success(f"✅ ผู้ขาย: พบในระบบ ({erp_vendor['name']})")
                
                    # เช็กยอดเงิน
                    if abs(f_total - erp_vendor["po_amount"]) < 0.01:
                        st.info(f"💰 ยอดเงิน: {f_total:,.2f} ตรงตามใบสั่งซื้อ (PO Match)")
                    else:
                        st.error(f"❌ ยอดเงิน: ไม่ตรง! (ในระบบตั้งเบิกไว้ {erp_vendor['po_amount']:,.2f})")
                else:
                    st.warning("❓ ผู้ขาย: ไม่พบเลขผู้เสียภาษีนี้ในฐานข้อมูล Vendor")

                if st.form_submit_button("✅ ยืนยันข้อมูลลงระบบ"):
                    if "saved_tables" not in st.session_state:
                        st.session_state["saved_tables"] = []
                    
                    if isinstance(edited_items, pd.DataFrame):
                        final_df = edited_items.copy()
                    else:
                        final_df = pd.DataFrame(edited_items)
                        
                    st.session_state["saved_tables"].append({
                        "header": {
                            "Invoice": f_inv,
                            "Date": f_date,
                            "Seller": f_seller,
                            "Total": f_total
                        },
                        "items": final_df
                    })
                    
                    st.balloons()
                    st.success("บันทึกสำเร็จ (ข้อมูลถูกส่งลงตารางด้านล่างแล้ว!)")

    else:
        # 💡 ข้อความต้อนรับตอนที่ยังไม่ได้อัปโหลดรูป
        st.info("👉 กรุณาอัปโหลดรูปใบเสร็จเพื่อเริ่มต้นดึงข้อมูล")

    # ==========================================
    # 📊 ส่วนแสดงตารางบันทึกข้อมูลรวม (จะแสดงเสมอถ้ามีข้อมูล ไม่ว่าจะลบรูปไปแล้วหรือไม่)
    # ==========================================
    if "saved_tables" in st.session_state and len(st.session_state["saved_tables"]) > 0:
        st.divider()
        st.subheader("📊 ตารางบันทึกข้อมูลรวม (เตรียมส่งออก Excel)")
        
        all_rows = []
        
        for record in st.session_state["saved_tables"]:
            h = record["header"]
            items_df = record["items"]
            
            if not items_df.empty:
                for _, row in items_df.iterrows():
                    all_rows.append({
                        "เลขที่ใบกำกับภาษี": h["Invoice"],
                        "วันที่": h["Date"],
                        "บริษัทผู้ขาย": h["Seller"],
                        "รายการสินค้า": row.get("name", ""),
                        "จำนวน": row.get("qty", 0),
                        "ราคา/หน่วย": row.get("unit_price", 0.0),
                        "ราคารวม": row.get("total", 0.0),
                        "ยอดสุทธิทั้งบิล": h["Total"]
                    })
            else:
                all_rows.append({
                    "เลขที่ใบกำกับภาษี": h["Invoice"],
                    "วันที่": h["Date"],
                    "บริษัทผู้ขาย": h["Seller"],
                    "รายการสินค้า": "ไม่ระบุรายการ",
                    "จำนวน": 1,
                    "ราคา/หน่วย": h["Total"],
                    "ราคารวม": h["Total"],
                    "ยอดสุทธิทั้งบิล": h["Total"]
                })
                
        summary_df = pd.DataFrame(all_rows)
        st.dataframe(summary_df, use_container_width=True)
        
        csv = summary_df.to_csv(index=False).encode('utf-8-sig')
        
        col_space, col_btn = st.columns([8, 2])
        with col_btn:
            st.download_button(
                label="📥 ดาวน์โหลด CSV",
                data=csv,
                file_name="mango_receipt_records.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    render_ui()