"""Integrated Pest Management (IPM) and Agronomic Dosage Calculator for Maize Diseases."""

DISEASE_PROFILES = {
    "Blight": {
        "common_name": "Northern Corn Leaf Blight (NCLB)",
        "pathogen": "Setosphaeria turcica (anamorph: Exserohilum turcicum)",
        "symptoms": "Cigar-shaped or elliptical grayish-green to tan lesions (2.5 to 15 cm long) on lower leaves progressing up the canopy.",
        "favorable_conditions": "Moderate temperatures (18–27°C / 64–81°F) with prolonged dew periods (6–8 hours) and overcast days.",
        "cultural_controls": [
            "Rotate out of maize for 1–2 years using non-host crops (soybean, groundnut, cowpea).",
            "Deep-till or shred crop residue post-harvest to accelerate fungal decomposition.",
            "Select hybrids with high polygenic resistance (Ht1, Ht2, or HtN resistance genes).",
            "Optimize plant density to avoid excessive canopy humidity and air stagnation."
        ],
        "biological_controls": [
            "Foliar spray of Trichoderma harzianum or Bacillus subtilis (5g/L water) at first sign of lesions.",
            "Neem Seed Kernel Extract (NSKE 5%) as an organic antifungal preventative."
        ],
        "chemical_treatments": [
            {
                "trade_name": "Azoxystrobin + Difenoconazole (e.g. Amistar Top)",
                "active_ingredient": "Strobilurin (QoI) + Triazole (DMI)",
                "rate_per_acre_ml": 200,
                "water_volume_per_acre_liters": 200,
                "phi_days": 21,
                "rei_hours": 12,
                "action_type": "Systemic preventive & curative"
            },
            {
                "trade_name": "Mancozeb 75% WP",
                "active_ingredient": "Dithiocarbamate",
                "rate_per_acre_ml": 600,  # 600 grams
                "water_volume_per_acre_liters": 200,
                "phi_days": 30,
                "rei_hours": 24,
                "action_type": "Broad-spectrum contact protectant"
            }
        ]
    },
    "Common_Rust": {
        "common_name": "Common Maize Rust",
        "pathogen": "Puccinia sorghi",
        "symptoms": "Small, powdery, circular to elongate golden-brown to cinnamon-colored pustules on both upper and lower leaf surfaces.",
        "favorable_conditions": "Cool to moderate temperatures (16–25°C / 60–77°F) with high relative humidity (>95%) and night dew.",
        "cultural_controls": [
            "Plant resistant maize hybrids possessing Rp gene resistance.",
            "Plant early in the season to avoid peak spore migration windows.",
            "Eradicate nearby alternative hosts (such as Oxalis / wood sorrel species).",
            "Maintain balanced nitrogen fertilization (excessive N promotes lush, susceptible tissue)."
        ],
        "biological_controls": [
            "Biological foliar formulation of Ampelomyces quisqualis or Bacillus amyloliquefaciens.",
            "Copper hydroxide (organic certified copper formulations) applied at onset."
        ],
        "chemical_treatments": [
            {
                "trade_name": "Tebuconazole 250 EC (e.g. Folicur)",
                "active_ingredient": "Demethylation Inhibitor (DMI Triazole)",
                "rate_per_acre_ml": 250,
                "water_volume_per_acre_liters": 200,
                "phi_days": 21,
                "rei_hours": 12,
                "action_type": "Systemic curative & anti-sporulant"
            },
            {
                "trade_name": "Pyraclostrobin (e.g. Headline)",
                "active_ingredient": "Strobilurin (QoI)",
                "rate_per_acre_ml": 180,
                "water_volume_per_acre_liters": 200,
                "phi_days": 14,
                "rei_hours": 12,
                "action_type": "Strong preventive & plant health enhancer"
            }
        ]
    },
    "Gray_Leaf_Spot": {
        "common_name": "Gray Leaf Spot (GLS)",
        "pathogen": "Cercospora zeae-maydis",
        "symptoms": "Narrow, rectangular, tan-to-gray lesions restricted by leaf veins (giving blocky, parallel-edged appearance).",
        "favorable_conditions": "Warm (25–32°C / 77–90°F) and humid conditions (>90% RH) with frequent rain or overhead irrigation.",
        "cultural_controls": [
            "Minimum 2-year crop rotation out of corn in no-till or conservation tillage fields.",
            "Bury infected stalks via conservation plowing where erosion permits.",
            "Utilize GLS-tolerant commercial hybrids.",
            "Avoid excessive late-afternoon overhead irrigation that leaves foliage wet overnight."
        ],
        "biological_controls": [
            "Bacillus subtilis strain QST 713 foliar spray.",
            "Potassium silicate bio-fortifier to strengthen epidermal leaf cell walls against fungal penetration."
        ],
        "chemical_treatments": [
            {
                "trade_name": "Fluxapyroxad + Pyraclostrobin (e.g. Priaxor)",
                "active_ingredient": "SDHI + QoI Dual Action",
                "rate_per_acre_ml": 150,
                "water_volume_per_acre_liters": 200,
                "phi_days": 21,
                "rei_hours": 12,
                "action_type": "High-potency systemic protectant & early curative"
            },
            {
                "trade_name": "Propiconazole 25% EC (e.g. Tilt)",
                "active_ingredient": "Triazole (DMI)",
                "rate_per_acre_ml": 200,
                "water_volume_per_acre_liters": 200,
                "phi_days": 30,
                "rei_hours": 24,
                "action_type": "Broad-spectrum systemic curative"
            }
        ]
    },
    "Healthy": {
        "common_name": "Healthy Maize Foliage",
        "pathogen": "None (No significant pathogen detected)",
        "symptoms": "Vibrant, uniform green foliage with intact leaf architecture and normal venation.",
        "favorable_conditions": "Optimal agronomic management: soil pH 6.0–6.8, balanced N-P-K, adequate moisture, good drainage.",
        "cultural_controls": [
            "Maintain bi-weekly scouting in a 'W' or 'Z' field walking pattern.",
            "Maintain soil organic matter and balanced micronutrient nutrition (Zinc, Boron, Magnesium).",
            "Employ drip or furrow irrigation rather than overhead sprinklers where feasible.",
            "Keep field borders free of weeds that harbor foliar pathogens."
        ],
        "biological_controls": [
            "Apply mycorrhizal fungi at planting to improve root vigor and natural systemic resistance (ISR)."
        ],
        "chemical_treatments": []
    }
}


def calculate_spray_dosage(disease_name, field_size_acres, tank_capacity_liters=16, treatment_index=0):
    """
    Calculates precise field-scale fungicide requirements and sprayer tank mixes.
    Assumes standard knapsack sprayer (16L default) or tractor-mounted rig.
    """
    if disease_name not in DISEASE_PROFILES or disease_name == "Healthy":
        return None

    treatments = DISEASE_PROFILES[disease_name]["chemical_treatments"]
    if not treatments:
        return None

    treatment = treatments[min(treatment_index, len(treatments) - 1)]
    
    total_water_liters = field_size_acres * treatment["water_volume_per_acre_liters"]
    total_chemical_qty = field_size_acres * treatment["rate_per_acre_ml"]
    number_of_tanks = total_water_liters / max(tank_capacity_liters, 1)
    chemical_per_tank = total_chemical_qty / max(number_of_tanks, 1)

    return {
        "product_name": treatment["trade_name"],
        "active_ingredient": treatment["active_ingredient"],
        "action_type": treatment["action_type"],
        "field_size_acres": field_size_acres,
        "total_water_liters": round(total_water_liters, 1),
        "total_chemical_amount": round(total_chemical_qty, 1),
        "unit": "ml" if "EC" in treatment["trade_name"] or "Top" in treatment["trade_name"] or "Priaxor" in treatment["trade_name"] or "Headline" in treatment["trade_name"] else "grams",
        "tank_capacity_liters": tank_capacity_liters,
        "total_tanks": round(number_of_tanks, 1),
        "dose_per_tank": round(chemical_per_tank, 1),
        "pre_harvest_interval_days": treatment["phi_days"],
        "re_entry_interval_hours": treatment["rei_hours"],
        "safety_notes": [
            "Wear Nitrile chemical-resistant gloves, eye goggles, and an N95/particulate respirator.",
            "Spray in early morning (6:00–9:00 AM) or late afternoon (4:00–6:30 PM) to avoid thermal updrafts and bee activity.",
            "Do not spray if wind speed exceeds 12 km/h (7.5 mph) or if heavy rain is expected within 3 hours."
        ]
    }
