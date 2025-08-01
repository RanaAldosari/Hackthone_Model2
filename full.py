import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FullInput(BaseModel):
    income: int
    commitments: int
    delay_in_sama: int
    job_type: str
    months_in_job: int
    account_type: str
    age: int
    principal: int
    term_months: int

def suggest_products(eligibility_percent, commitment_ratio, age, months_in_job):
    products = [
        "بطاقة ائتمانية Infinite",
        "تمويل شخصي مرابحة",
        "تمويل عقاري",
        "بطاقة ائتمانية Signature",
        "حساب توفير",
        "تمويل سيارة",
        "صكوك استثمارية",
        "صناديق استثمارية"
    ]
    if eligibility_percent > 80:
        return products
    elif eligibility_percent > 50:
        return products[:4]  
    else:
        reasons = []
        if commitment_ratio > 0.4:
            reasons.append("نسبة الالتزامات عالية (أكثر من 40% من الدخل)")
        if age < 30:
            reasons.append("العمر أقل من 30 سنة")
        if months_in_job < 12:
            reasons.append("مدة العمل أقل من 12 شهرًا")
        reason_str = "؛ ".join(reasons) if reasons else "سبب غير محدد - تحقق من البيانات"
        return f"غير مؤهل بسبب: {reason_str}"

def suggest_investment(surplus, eligibility_percent):
    investments = [
        "حساب توفير ( 2%)",
        "صكوك شرعية ( 3-4%)",
        "صناديق عقارية ( 5%+)",
        "استثمار في أسهم ( 10%)",
        "حساب جاري ( 1%)",
        "صناديق مالية ( 4%)",
        "استثمار في ذهب (مخاطر متوسطة)",
        "صناديق دولية ( 6%+)"
    ]
    if surplus > 5000 and eligibility_percent > 80:
        return investments
    elif surplus > 1000 and eligibility_percent > 50:
        return investments[:4]
    else:
        return "غير كافي - ركز على التوفير"

@app.post("/predict_full")
def predict_full(data: FullInput):

    with open('full_integrated_model.pkl', 'rb') as f:
        models = pickle.load(f)

    eligibility_model = models["eligibility"]
    payment_model = models["payment"]
    surplus_model = models["surplus"]

    # تنبؤ التأهيل
    el_df = pd.DataFrame({
        "الدخل": [data.income],
        "الالتزامات": [data.commitments],
        "تأخير في سمة": [data.delay_in_sama],
        "نوع الوظيفة": [data.job_type],
        "مدة العمل": [data.months_in_job],
        "نوع الحساب": [data.account_type],
        "العمر": [data.age]
    })
    eligibility_percent = round(eligibility_model.predict(el_df)[0], 1)

    # نسبة الالتزامات
    commitment_ratio = data.commitments / data.income if data.income > 0 else 0

    # تنبؤ الدفعة الشهرية
    pay_df = pd.DataFrame({
        "المبلغ": [data.principal],
        "مدة السداد": [data.term_months],
        "العمر": [data.age],
        "الدخل": [data.income],
        "نسبة التأهيل": [eligibility_percent]
    })
    monthly_payment = round(payment_model.predict(pay_df)[0], 1)

    # تنبؤ الفائض
    sur_df = pd.DataFrame({
        "الدخل": [data.income],
        "الالتزامات": [data.commitments],
        "الدفعة الشهرية": [monthly_payment]
    })
    surplus = round(surplus_model.predict(sur_df)[0], 1)

    # التوصيات
    products = suggest_products(eligibility_percent, commitment_ratio, data.age, data.months_in_job)
    investment = suggest_investment(surplus, eligibility_percent)

    return {
        "نسبة التأهيل": eligibility_percent,
        "المنتجات المقترحة": products,
        "الفائض المالي": surplus,
        "خيار الاستثمار": investment
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
