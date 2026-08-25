"""
Disease knowledge base.

Started from the notebook's `health_data` dict (cell 55), which only
covered 4 of the model's 41 diseases (Fungal infection, Common Cold,
Malaria, Diabetes) — those 4 entries are kept exactly as the notebook
wrote them. All other 37 diseases the model can predict have been added
here with the same structure (general description, general dietary
guidance, general precautions), so `disease_details()` no longer falls
back to "information not available" for most predictions.

This is general educational information only, not medical advice — see
the disclaimer shown in the UI. Keys are the disease names as they appear
in Training.csv/Testing.csv, stripped of leading/trailing whitespace
(a few labels in the source dataset have trailing spaces, e.g.
"Diabetes ", "Hypertension " — disease_details() strips its input before
looking it up, so keys here are stored stripped too).
"""

health_data = {
    "(vertigo) Paroymsal  Positional Vertigo": {
        "description": "A common inner-ear condition causing brief, intense episodes of "
        "dizziness triggered by specific changes in head position.",
        "diet": "Stay well hydrated and eat regular, balanced meals; avoid excess caffeine and alcohol.",
        "precautions": [
            "Sit or lie down immediately when dizziness starts",
            "Avoid sudden head movements",
            "Get up slowly from lying or sitting positions",
            "Consult a doctor for balance exercises or evaluation",
        ],
    },
    "AIDS": {
        "description": "The advanced stage of HIV infection, in which the immune system is "
        "severely weakened and vulnerable to infections.",
        "diet": "High-protein, nutrient-dense meals to support the immune system, with food handled hygienically.",
        "precautions": [
            "Take antiretroviral therapy exactly as prescribed",
            "Practice safe sex and avoid sharing needles",
            "Attend regular medical checkups",
            "Seek prompt care for any new infection",
        ],
    },
    "Acne": {
        "description": "A common skin condition where hair follicles become clogged with oil "
        "and dead skin cells, causing pimples and blackheads.",
        "diet": "Limit high-sugar and high-glycemic foods; stay hydrated and eat plenty of vegetables.",
        "precautions": [
            "Wash the affected area gently, twice a day",
            "Avoid picking or popping pimples",
            "Use non-comedogenic skincare products",
            "See a dermatologist if it's severe or scarring",
        ],
    },
    "Alcoholic hepatitis": {
        "description": "Liver inflammation caused by heavy, prolonged alcohol consumption.",
        "diet": "A balanced, protein-adequate diet; complete alcohol avoidance is essential.",
        "precautions": [
            "Stop drinking alcohol completely",
            "Get regular liver function monitoring",
            "Take prescribed medications and vitamins as advised",
            "Seek support for alcohol use if needed",
        ],
    },
    "Allergy": {
        "description": "An overreaction of the immune system to a normally harmless substance, "
        "such as pollen, dust, or certain foods.",
        "diet": "Avoid known trigger foods; otherwise a normal balanced diet is fine.",
        "precautions": [
            "Identify and avoid known allergens",
            "Keep antihistamines on hand if prescribed",
            "Keep living spaces clean and well-ventilated",
            "Seek emergency care for any difficulty breathing or swelling",
        ],
    },
    "Arthritis": {
        "description": "Inflammation of one or more joints, causing pain, swelling, and "
        "stiffness that can worsen with age or activity.",
        "diet": "Anti-inflammatory foods such as fatty fish, fruits, and vegetables; limit processed food.",
        "precautions": [
            "Stay physically active with joint-friendly exercise",
            "Maintain a healthy weight to reduce joint stress",
            "Use hot/cold therapy for flare-ups as advised",
            "Consult a doctor about pain management options",
        ],
    },
    "Bronchial Asthma": {
        "description": "A chronic condition in which the airways narrow and swell, causing "
        "wheezing, shortness of breath, and coughing.",
        "diet": "A balanced diet rich in fruits and vegetables; identify and avoid any food triggers.",
        "precautions": [
            "Keep prescribed inhalers accessible at all times",
            "Avoid known triggers like smoke, dust, and strong odors",
            "Follow your asthma action plan",
            "Seek emergency care for severe breathing difficulty",
        ],
    },
    "Cervical spondylosis": {
        "description": "Age-related wear and tear affecting the spinal discs and joints in the neck.",
        "diet": "A balanced diet with adequate calcium and vitamin D for bone health.",
        "precautions": [
            "Maintain good posture, especially when using screens",
            "Do gentle neck stretches and strengthening exercises",
            "Avoid carrying heavy loads on the shoulders",
            "Consult a doctor or physiotherapist for persistent pain",
        ],
    },
    "Chicken pox": {
        "description": "A highly contagious viral infection causing an itchy, blister-like rash, "
        "most common in childhood.",
        "diet": "Soft, easy-to-swallow foods and plenty of fluids, especially if mouth sores are present.",
        "precautions": [
            "Stay isolated until all blisters have crusted over",
            "Keep fingernails short to avoid scratching",
            "Use calamine lotion or prescribed treatment for itching",
            "Consult a doctor, especially for adults or high-risk groups",
        ],
    },
    "Chronic cholestasis": {
        "description": "A condition where bile flow from the liver is reduced or blocked, "
        "leading to jaundice and itching.",
        "diet": "A low-fat diet; a doctor may recommend fat-soluble vitamin supplementation.",
        "precautions": [
            "Follow up regularly with a liver specialist",
            "Avoid alcohol and unnecessary medications",
            "Report worsening jaundice or itching promptly",
            "Take prescribed medication for itch relief as directed",
        ],
    },
    "Common Cold": {
        "description": "A viral infection affecting the upper respiratory tract.",
        "diet": "Warm soup, fruits, fluids and light meals.",
        "precautions": [
            "Drink warm water",
            "Take proper rest",
            "Wash hands regularly",
            "Avoid cold drinks",
        ],
    },
    "Dengue": {
        "description": "A mosquito-borne viral infection causing high fever, severe body aches, "
        "and fatigue.",
        "diet": "Plenty of fluids, papaya leaf extract (as commonly advised), and light, nutritious meals.",
        "precautions": [
            "Rest and stay well hydrated",
            "Avoid mosquito bites with nets and repellents",
            "Monitor for warning signs like bleeding or severe abdominal pain",
            "Seek immediate medical care if symptoms worsen",
        ],
    },
    "Diabetes": {
        "description": "A chronic disease affecting blood glucose levels.",
        "diet": "High fiber diet and low sugar intake.",
        "precautions": [
            "Exercise daily",
            "Monitor blood sugar",
            "Take medicines regularly",
            "Avoid sugary foods",
        ],
    },
    "Dimorphic hemmorhoids(piles)": {
        "description": "Swollen veins in the lower rectum or anus, often causing pain, itching, "
        "and bleeding during bowel movements.",
        "diet": "High-fiber foods and plenty of water to soften stools and ease bowel movements.",
        "precautions": [
            "Avoid straining during bowel movements",
            "Stay physically active",
            "Use sitz baths for relief",
            "Consult a doctor if bleeding persists or is severe",
        ],
    },
    "Drug Reaction": {
        "description": "An adverse or allergic reaction of the body to a medication.",
        "diet": "No specific diet; stay hydrated and eat light, easily digestible food while recovering.",
        "precautions": [
            "Stop the suspected medication and contact a doctor",
            "Note down the medication that caused the reaction",
            "Avoid that medication (and similar ones) in future",
            "Seek emergency care for severe swelling or breathing difficulty",
        ],
    },
    "Fungal infection": {
        "description": "A fungal infection affects the skin, nails or hair.",
        "diet": "Drink plenty of water and consume Vitamin C rich foods.",
        "precautions": [
            "Keep skin clean and dry",
            "Avoid sharing towels",
            "Use antifungal medication",
            "Consult a dermatologist",
        ],
    },
    "GERD": {
        "description": "Gastroesophageal reflux disease — stomach acid frequently flows back "
        "into the esophagus, causing heartburn and discomfort.",
        "diet": "Smaller meals, avoiding spicy/fatty/acidic foods, and not lying down right after eating.",
        "precautions": [
            "Avoid lying down immediately after meals",
            "Reduce intake of caffeine, alcohol, and spicy food",
            "Maintain a healthy weight",
            "Consult a doctor if symptoms persist despite lifestyle changes",
        ],
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines, usually causing diarrhea, "
        "vomiting, and abdominal cramps.",
        "diet": "Clear fluids, ORS, and bland foods (like rice, bananas, toast) as tolerated.",
        "precautions": [
            "Stay hydrated with fluids and oral rehydration solution",
            "Rest and eat light, bland food as tolerated",
            "Practice good hand hygiene to avoid spreading it",
            "Seek care if unable to keep fluids down or if symptoms are severe",
        ],
    },
    "Heart attack": {
        "description": "A blockage of blood flow to part of the heart muscle — a medical emergency.",
        "diet": "A heart-healthy, low-sodium, low-saturated-fat diet as part of long-term recovery.",
        "precautions": [
            "Call emergency services immediately for chest pain",
            "Do not drive yourself to the hospital",
            "Follow prescribed cardiac medication and rehab plans",
            "Avoid smoking and manage blood pressure/cholesterol long-term",
        ],
    },
    "Hepatitis B": {
        "description": "A viral infection that causes liver inflammation, spread through blood "
        "and body fluids.",
        "diet": "A balanced, liver-friendly diet; avoid alcohol.",
        "precautions": [
            "Get vaccinated if not already immune",
            "Avoid sharing needles or personal items like razors",
            "Practice safe sex",
            "Get regular liver monitoring from a doctor",
        ],
    },
    "Hepatitis C": {
        "description": "A viral infection causing liver inflammation, spread mainly through "
        "contact with infected blood.",
        "diet": "A balanced, liver-friendly diet; avoid alcohol.",
        "precautions": [
            "Avoid sharing needles or personal items like razors",
            "Complete the full course of antiviral treatment if prescribed",
            "Get regular liver function monitoring",
            "Avoid alcohol to protect the liver",
        ],
    },
    "Hepatitis D": {
        "description": "A viral liver infection that only occurs in people already infected "
        "with Hepatitis B.",
        "diet": "A balanced, liver-friendly diet; avoid alcohol.",
        "precautions": [
            "Get vaccinated against Hepatitis B to prevent this co-infection",
            "Avoid alcohol and unnecessary medications",
            "Attend regular liver checkups",
            "Follow the treatment plan prescribed by a specialist",
        ],
    },
    "Hepatitis E": {
        "description": "A viral liver infection usually spread through contaminated water.",
        "diet": "Plenty of fluids and a light, liver-friendly diet during recovery.",
        "precautions": [
            "Drink only clean, safe water",
            "Practice good food and hand hygiene",
            "Rest and stay hydrated during recovery",
            "Seek medical care promptly, especially if pregnant",
        ],
    },
    "Hypertension": {
        "description": "Persistently high blood pressure in the arteries, often with no "
        "obvious symptoms, that raises the risk of heart disease and stroke.",
        "diet": "A low-sodium diet rich in fruits, vegetables, and whole grains (e.g. DASH-style eating).",
        "precautions": [
            "Monitor blood pressure regularly",
            "Reduce salt intake",
            "Exercise regularly and manage weight",
            "Take prescribed medication consistently",
        ],
    },
    "Hyperthyroidism": {
        "description": "A condition where the thyroid gland produces excess thyroid hormone, "
        "speeding up the body's metabolism.",
        "diet": "Adequate calories and calcium; a doctor may advise limiting iodine-rich foods.",
        "precautions": [
            "Take prescribed anti-thyroid medication as directed",
            "Attend regular thyroid function monitoring",
            "Manage stress, which can worsen symptoms",
            "Report rapid heartbeat or significant weight loss to a doctor",
        ],
    },
    "Hypoglycemia": {
        "description": "Abnormally low blood sugar levels, which can cause shakiness, "
        "confusion, and fainting if untreated.",
        "diet": "Regular, balanced meals with complex carbohydrates; avoid skipping meals.",
        "precautions": [
            "Carry a fast-acting sugar source (like glucose tablets or juice)",
            "Eat regular meals and snacks; avoid skipping them",
            "Monitor blood sugar if you have diabetes",
            "Seek medical advice if episodes are frequent",
        ],
    },
    "Hypothyroidism": {
        "description": "A condition where the thyroid gland doesn't produce enough thyroid "
        "hormone, slowing the body's metabolism.",
        "diet": "A balanced diet with adequate iodine; consult a doctor about iodine intake if on medication.",
        "precautions": [
            "Take prescribed thyroid hormone medication consistently",
            "Get thyroid levels checked regularly",
            "Report ongoing fatigue or weight changes to a doctor",
            "Take medication on an empty stomach as usually advised",
        ],
    },
    "Impetigo": {
        "description": "A contagious bacterial skin infection causing red sores that can "
        "burst and develop a honey-colored crust.",
        "diet": "No specific diet; a balanced diet supports overall healing.",
        "precautions": [
            "Keep the affected area clean and covered",
            "Avoid scratching and touching the sores",
            "Wash hands frequently and avoid sharing towels",
            "Use prescribed antibiotic treatment as directed",
        ],
    },
    "Jaundice": {
        "description": "Yellowing of the skin and eyes caused by elevated bilirubin, often "
        "linked to liver problems.",
        "diet": "A low-fat, liver-friendly diet with plenty of fluids; avoid alcohol.",
        "precautions": [
            "Rest and stay hydrated",
            "Avoid alcohol and unnecessary medications",
            "Identify and treat the underlying cause with a doctor",
            "Get liver function tests as advised",
        ],
    },
    "Malaria": {
        "description": "A mosquito-borne infectious disease.",
        "diet": "Drink ORS, coconut water and eat nutritious food.",
        "precautions": [
            "Use mosquito nets",
            "Complete medication course",
            "Stay hydrated",
            "Consult doctor immediately",
        ],
    },
    "Migraine": {
        "description": "A recurring headache disorder causing throbbing pain, often with "
        "nausea and sensitivity to light or sound.",
        "diet": "Stay hydrated and identify/avoid personal food triggers (common ones include caffeine and alcohol).",
        "precautions": [
            "Rest in a quiet, dark room during an attack",
            "Identify and avoid personal migraine triggers",
            "Maintain regular sleep and meal schedules",
            "Consult a doctor for frequent or severe migraines",
        ],
    },
    "Osteoarthristis": {
        "description": "A degenerative joint disease caused by the breakdown of cartilage, "
        "leading to pain and stiffness, most common in older adults.",
        "diet": "Anti-inflammatory foods and adequate calcium and vitamin D; maintain a healthy weight.",
        "precautions": [
            "Stay active with low-impact exercise like swimming or walking",
            "Maintain a healthy weight to reduce joint load",
            "Use supportive footwear and joint aids as needed",
            "Consult a doctor about pain management options",
        ],
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Loss of muscle function caused by bleeding in the brain — a serious "
        "medical emergency requiring immediate care.",
        "diet": "As advised by medical staff during recovery; often requires swallowing-safe textures.",
        "precautions": [
            "Seek emergency medical care immediately",
            "Do not give food or water if swallowing is affected",
            "Follow the prescribed rehabilitation program closely",
            "Manage underlying risk factors like high blood pressure",
        ],
    },
    "Peptic ulcer diseae": {
        "description": "Open sores that develop on the lining of the stomach or upper small "
        "intestine, often causing burning stomach pain.",
        "diet": "Smaller, frequent meals; avoid spicy, acidic, and fried foods, and alcohol.",
        "precautions": [
            "Avoid NSAIDs (like ibuprofen) unless advised otherwise by a doctor",
            "Limit alcohol and spicy or acidic foods",
            "Avoid smoking",
            "Complete any prescribed antibiotic/acid-reducing treatment fully",
        ],
    },
    "Pneumonia": {
        "description": "An infection that inflames the air sacs in one or both lungs, which "
        "may fill with fluid, causing cough and difficulty breathing.",
        "diet": "Plenty of fluids and light, nutritious meals to support recovery.",
        "precautions": [
            "Complete the full course of prescribed antibiotics if bacterial",
            "Rest and stay well hydrated",
            "Avoid smoking and secondhand smoke",
            "Seek immediate care for severe breathing difficulty",
        ],
    },
    "Psoriasis": {
        "description": "A chronic autoimmune condition that speeds up skin cell turnover, "
        "causing thick, scaly patches.",
        "diet": "An anti-inflammatory diet; some people find limiting alcohol and processed food helps.",
        "precautions": [
            "Keep skin moisturized",
            "Avoid known triggers like stress or skin injury",
            "Get moderate, safe sun exposure if advised by a doctor",
            "Follow prescribed topical or systemic treatment",
        ],
    },
    "Tuberculosis": {
        "description": "A bacterial infection that mainly affects the lungs and spreads "
        "through the air when an infected person coughs.",
        "diet": "High-protein, calorie-dense meals to support recovery and weight maintenance.",
        "precautions": [
            "Complete the full course of prescribed antibiotics, even if feeling better",
            "Cover your mouth when coughing and ensure good ventilation",
            "Attend all follow-up appointments",
            "Avoid close contact with others until cleared by a doctor",
        ],
    },
    "Typhoid": {
        "description": "A bacterial infection spread through contaminated food or water, "
        "causing prolonged fever and stomach problems.",
        "diet": "Soft, easily digestible, high-calorie food and plenty of fluids.",
        "precautions": [
            "Drink only clean, safe water",
            "Complete the full course of prescribed antibiotics",
            "Practice good hand and food hygiene",
            "Rest and stay hydrated during recovery",
        ],
    },
    "Urinary tract infection": {
        "description": "An infection in any part of the urinary system, most often the "
        "bladder, causing pain and a frequent urge to urinate.",
        "diet": "Plenty of water; cranberry juice is commonly suggested, though evidence is mixed.",
        "precautions": [
            "Drink plenty of water",
            "Urinate frequently and avoid holding it in",
            "Practice good hygiene",
            "Complete the full course of prescribed antibiotics",
        ],
    },
    "Varicose veins": {
        "description": "Swollen, twisted veins, usually in the legs, caused by weakened vein "
        "valves and poor blood flow.",
        "diet": "A high-fiber, low-salt diet to support circulation and avoid constipation.",
        "precautions": [
            "Avoid standing or sitting for long periods without moving",
            "Elevate your legs when resting",
            "Wear compression stockings if advised",
            "Stay physically active",
        ],
    },
    "hepatitis A": {
        "description": "A viral liver infection usually spread through contaminated food or water.",
        "diet": "Plenty of fluids and a light, liver-friendly diet during recovery.",
        "precautions": [
            "Get vaccinated if at risk",
            "Practice good hand and food hygiene",
            "Drink only clean, safe water",
            "Rest and avoid alcohol during recovery",
        ],
    },
}


def disease_details(disease):
    """
    Reproduces the notebook's disease_details() (cell 56), but returns a
    dict instead of printing, so the Streamlit UI can render it. Falls
    back to None fields (not fabricated text) if a disease somehow isn't
    in the knowledge base, exactly as the notebook's "Information not
    available" branch did.
    """
    disease_key = disease.strip() if isinstance(disease, str) else disease

    if disease_key in health_data:
        entry = health_data[disease_key]
        return {
            "available": True,
            "description": entry["description"],
            "diet": entry["diet"],
            "precautions": entry["precautions"],
        }

    return {
        "available": False,
        "description": None,
        "diet": None,
        "precautions": None,
    }
