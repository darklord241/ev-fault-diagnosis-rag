import httpx
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.common.models import EvidenceSource, DTCRecord, ComponentSpec
from src.common.logging import logger
from config.settings import settings

# Note
SAMPLE_EV_EVIDENCE_STORE: List[Dict[str, Any]] = [
    {
        "source_id": "SRC-EV-BMS-001",
        "document_title": "Electric Two-Wheeler High-Voltage Battery & BMS Service Guide",
        "document_type": "Service Manual",
        "section": "Battery Management System - Cell Imbalance & Undervoltage Protection",
        "page_number": 42,
        "vehicle_model": "Generic 48V-72V Electric Scooter / Ather / Ola",
        "subsystem": "Battery Management System (BMS)",
        "content": (
            "Symptom: Vehicle loses power abruptly under acceleration; dashboard indicates sudden state-of-charge (SOC) drop from 40% to 5%.\n"
            "Diagnostic Logic:\n"
            "1. Cell Voltage Delta: Check individual cell voltages using diagnostic tool. If delta V between highest and lowest cell exceeds 150mV (0.15V), BMS initiates low-voltage protection cut-off under load.\n"
            "2. Thermal Sensor Fault: NTC thermistor resistance at 25°C should be 10kΩ ± 5%. High resistance (>50kΩ) triggers false over-temperature protection.\n"
            "3. Action: Perform slow balanced charging cycle (CV mode at 0.1C) for 12 hours. If delta remains >200mV, replace degraded cell module. Never bypass BMS discharge MOSFETs."
        ),
        "relevance_score": 0.95
    },
    {
        "source_id": "SRC-EV-MCU-002",
        "document_title": "PMSM / BLDC Motor Controller Field Diagnostic Handbook",
        "document_type": "Field Engineering Manual",
        "section": "Inverter Stage & Hall Effect Angle Sensor Troubleshooting",
        "page_number": 18,
        "vehicle_model": "All 2-Wheeler Hub & Mid-Drive PMSM Systems",
        "subsystem": "Motor Controller / Powertrain",
        "content": (
            "Symptom: Motor stutters, vibrates heavily at low RPM, or refuses to spin with throttle input while throwing Hall Sensor Error (Error Code E-08 / DTC P0A3F).\n"
            "Diagnostic Steps:\n"
            "1. Measure Hall sensor supply voltage: Pins 1 (VCC) and 2 (GND) must read 5.0V ± 0.2V DC.\n"
            "2. Measure Signal Outputs: Slowly rotate rear wheel by hand while measuring Hall signals (Ha, Hb, Hc) against GND. Voltage should alternate cleanly between 0.0V (logic LOW) and 4.8V-5.0V (logic HIGH).\n"
            "3. A constant 0V or 5V indicates a damaged Hall IC (typically SS41F or equivalent) or broken wire in the motor harness.\n"
            "4. Phase MOSFET Check: With battery disconnected, measure resistance between B+ and Phase U, V, W, then B- and Phase U, V, W. Resistance should read >50kΩ (reverse diode drop 0.45V-0.55V). Short circuit (<10Ω) confirms blown MOSFET."
        ),
        "relevance_score": 0.92
    },
    {
        "source_id": "SRC-EV-THR-003",
        "document_title": "Drive-by-Wire Throttle & Brake Interlock Diagnostic Manual",
        "document_type": "Technical Specification",
        "section": "Electronic Throttle (Hall Sensor) Ground Loss & High Idle Cutoff",
        "page_number": 12,
        "vehicle_model": "Electric Two-Wheeler Standard",
        "subsystem": "Throttle & Safety Controls",
        "content": (
            "Symptom: Vehicle does not respond to throttle or cuts off intermittently when handlebar is turned.\n"
            "Pinout Specification (3-pin waterproof connector):\n"
            "- Pin 1 (Red): +5V DC Supply from MCU\n"
            "- Pin 2 (Black): Ground (GND)\n"
            "- Pin 3 (Green/White): Throttle Signal Voltage (0.8V idle to 4.2V at full throttle)\n"
            "Safety Cutoff: If signal voltage is <0.5V (broken wire) or >4.5V (GND open circuit), MCU enters fail-safe lockout mode.\n"
            "Brake Switch Interlock: Electronic brake switches (NC or NO) ground the brake cut-off pin on MCU. If brake sensor is stuck closed or shorted, throttle is completely inhibited."
        ),
        "relevance_score": 0.89
    },
    {
        "source_id": "SRC-EV-CHG-004",
        "document_title": "On-Board & Off-Board Charger Interlock Guidelines",
        "document_type": "Service Manual",
        "section": "CC/CV Charging Handshake & Thermal Protection",
        "page_number": 29,
        "vehicle_model": "Electric 2-Wheeler Fast/Slow Charging Systems",
        "subsystem": "Charging Subsystem",
        "content": (
            "Symptom: Charger LED blinks red; battery will not charge or stops after 2 minutes.\n"
            "Protocol & Diagnostic Checks:\n"
            "1. Pilot / Communication Line: Check CP (Control Pilot) or CAN-H/CAN-L line for 120Ω termination resistance with power OFF.\n"
            "2. Pre-charge Voltage: Charger verifies pack open circuit voltage (OCV). If pack voltage is below 36V (on a 48V nominal system), charger locks out to prevent charging over-discharged cells.\n"
            "3. Charging Port Thermal Switch: If charging port temperature sensor detects >65°C, charging rate is throttled or aborted."
        ),
        "relevance_score": 0.85
    },
    {
        "source_id": "SRC-EV-DCDC-005",
        "document_title": "Auxiliary Electrical & DC-DC Converter Service Manual",
        "document_type": "Electrical Wiring Guide",
        "section": "12V Auxiliary Rail & Headlamp/Horn Failure",
        "page_number": 7,
        "vehicle_model": "Generic Electric Scooter",
        "subsystem": "Auxiliary Electrics",
        "content": (
            "Symptom: Traction drive operates normally, but 12V loads (headlamp, turn signals, horn, display backlight) are non-functional.\n"
            "Diagnostic Steps:\n"
            "1. Verify DC-DC converter input: Measure high voltage (48V-72V DC) at input terminals with ignition switch ON.\n"
            "2. If input voltage is present, check 12V output. Output must be 12.8V - 13.5V DC.\n"
            "3. Check internal inline fuse (typically 10A or 15A blade fuse) on the 12V output harness."
        ),
        "relevance_score": 0.81
    }
]

SAMPLE_DTC_DATABASE: Dict[str, Dict[str, Any]] = {
    "P0A80": {
        "code": "P0A80",
        "system": "BMS",
        "description": "Replace Hybrid / EV Battery Pack - Severe Cell Voltage Deviation",
        "symptoms": ["Limp home mode", "Sudden SOC drop", "Turtle icon on cluster"],
        "possible_causes": ["Degraded cell group", "BMS balancing circuit MOSFET failure", "Loose busbar bolt"],
        "suggested_checks": ["Measure cell voltages at BMS balance harness", "Inspect busbar contact resistance (<0.5mΩ)"],
        "source_manual": "EV High Voltage Diagnostic Standard ISO-26262 / OEM Workshop Manual p.88"
    },
    "E-08": {
        "code": "E-08",
        "system": "Motor Controller",
        "description": "BLDC Motor Hall Sensor Circuit Malfunction",
        "symptoms": ["Motor hesitation / shuddering", "No drive", "Motor growling sound"],
        "possible_causes": ["Burnt Hall sensor IC in stator", "Damaged wire harness near swingarm pivot", "Loose 5-pin waterproof connector"],
        "suggested_checks": ["Check 5V supply at pin 1", "Probe Ha, Hb, Hc while spinning wheel manually"],
        "source_manual": "PMSM/BLDC Controller Fault Code Reference 2024"
    },
    "E-07": {
        "code": "E-07",
        "system": "Throttle",
        "description": "Throttle Signal Voltage Out of Range (High/Low)",
        "symptoms": ["Throttle unresponsive", "Dashboard warning icon"],
        "possible_causes": ["Ground wire broken (reads >4.5V)", "Signal wire short to ground (reads 0V)", "Water ingress in throttle housing"],
        "suggested_checks": ["Measure voltage across Signal and Ground wires (nominal 0.8V - 4.2V)"],
        "source_manual": "Electric Two-Wheeler Drive-by-Wire Standard Manual p.15"
    },
    "U0100": {
        "code": "U0100",
        "system": "CAN Communication",
        "description": "Lost Communication with Motor Control Module (MCU)",
        "symptoms": ["Cluster shows dashes '--' for speed and motor temp", "Vehicle will not arm"],
        "possible_causes": ["CAN-H / CAN-L twisted pair cut", "Termination resistor (120Ω) missing", "MCU logic power fuse blown"],
        "suggested_checks": ["Check 60Ω resistance between CAN-H and CAN-L with key OFF", "Check 12V ignition feed to MCU"],
        "source_manual": "EV CAN Bus Troubleshooting Guide Vol. 2"
    }
}

GENERIC_DTC_DATABASE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "generic_dtc_codes.json"
)


def _load_generic_dtc_database() -> Dict[str, Dict[str, Any]]:
    try:
        records = json.loads(GENERIC_DTC_DATABASE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not load generic DTC database: %s", error)
        return {}
    return {
        record["code"].upper(): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("code"), str)
    }


GENERIC_DTC_DATABASE = _load_generic_dtc_database()

SAMPLE_COMPONENT_SPECS: Dict[str, Dict[str, Any]] = {
    "throttle": {
        "component_name": "Hall Effect Electronic Throttle",
        "subsystem": "Controls",
        "nominal_operating_specs": {
            "supply_voltage": "5.0V DC (±0.2V)",
            "idle_voltage_output": "0.85V ± 0.15V",
            "full_throttle_voltage_output": "4.15V ± 0.2V",
            "operating_temp": "-20°C to +65°C"
        },
        "pinout_details": {
            "Pin 1 (Red)": "+5V Power Supply",
            "Pin 2 (Black)": "Ground",
            "Pin 3 (Green)": "Output Signal"
        },
        "troubleshooting_thresholds": {
            "low_fault_threshold": "< 0.5V (Broken signal wire or disconnected)",
            "high_fault_threshold": "> 4.6V (GND open circuit condition)"
        },
        "source_reference": "Drive-by-Wire Component Standard Spec v3.2"
    },
    "hall_sensor": {
        "component_name": "Motor Stator Hall Effect Position Sensor Array (3-phase)",
        "subsystem": "Motor / Powertrain",
        "nominal_operating_specs": {
            "supply_voltage": "4.5V - 12V DC (typical 5V from controller)",
            "sensor_type": "Bipolar Latching Hall IC (SS41F / 41F)",
            "output_type": "Open-collector (pulled up to 5V inside MCU)",
            "phase_angle": "120° electrical spacing"
        },
        "pinout_details": {
            "Pin 1 (Red)": "+5V VCC",
            "Pin 2 (Black)": "GND",
            "Pin 3 (Yellow)": "Hall Phase A",
            "Pin 4 (Green)": "Hall Phase B",
            "Pin 5 (Blue)": "Hall Phase C"
        },
        "troubleshooting_thresholds": {
            "logic_low": "0.0V - 0.4V DC",
            "logic_high": "4.5V - 5.0V DC"
        },
        "source_reference": "BLDC/PMSM Hub Motor Repair Manual Section 4"
    },
    "bms": {
        "component_name": "Lithium-ion Smart Battery Management System",
        "subsystem": "Battery",
        "nominal_operating_specs": {
            "nominal_voltage": "48V / 60V / 72V configurations (13S - 20S Li-ion or 16S - 24S LFP)",
            "cell_overvoltage_cutoff": "4.25V per cell (NMC) / 3.65V per cell (LFP)",
            "cell_undervoltage_cutoff": "2.80V per cell (NMC) / 2.50V per cell (LFP)",
            "max_balancing_current": "50mA - 150mA passive bleed"
        },
        "pinout_details": {
            "B-": "Main Battery Negative Pack Terminal",
            "P- / C-": "Power Discharge & Charge Port Negative",
            "B1..B20": "Individual Cell Balancing Harness",
            "CAN_H / CAN_L": "Communication to Vehicle Master Control Unit"
        },
        "troubleshooting_thresholds": {
            "max_allowed_cell_delta": "100mV under resting state, 200mV under max acceleration",
            "temp_cutoff_high": "60°C discharge / 45°C charge"
        },
        "source_reference": "ISO-6469 & OEM EV Battery Architecture Spec"
    }
}

class BackendRetrievalAdapter:
    """
    Adapter layer connecting to the teammate's Backend API / Hybrid Retrieval pipeline.
    If the Backend API is not available or HYBRID_RETRIEVAL_USE_MOCK is true,
    it falls back seamlessly to the embedded EV diagnostic knowledge base.
    """

    def __init__(self):
        self.use_mock = settings.HYBRID_RETRIEVAL_USE_MOCK
        self.base_url = settings.HYBRID_RETRIEVAL_API_BASE_URL
        self.api_key = settings.HYBRID_RETRIEVAL_API_KEY
        logger.info(f"Initialized BackendRetrievalAdapter (Mock Mode: {self.use_mock}, URL: {self.base_url})")

    async def _fetch_live_evidence(
        self,
        query: str,
        vehicle_model: Optional[str],
        subsystem: Optional[str],
        top_k: int,
    ) -> List[EvidenceSource]:
        if not self.api_key:
            raise RuntimeError(
                "Live retrieval is enabled, but HYBRID_RETRIEVAL_API_KEY is not configured."
            )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/search",
                    json={
                        "query": query,
                        "vehicle_model": vehicle_model,
                        "subsystem": subsystem,
                        "top_k": top_k,
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Retrieval backend returned HTTP {response.status_code}."
                )
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise ValueError(
                    "Retrieval backend response must contain a results array."
                )
            return [
                EvidenceSource.model_validate(item)
                for item in data["results"]
            ]
        except RuntimeError:
            raise
        except (httpx.RequestError, ValueError, TypeError) as error:
            raise RuntimeError(
                "Live retrieval failed or returned an invalid response; "
                "sample evidence was not used."
            ) from error

    async def search_hybrid_evidence(
        self,
        query: str,
        vehicle_model: Optional[str] = None,
        subsystem: Optional[str] = None,
        top_k: int = 4
    ) -> List[EvidenceSource]:
        """
        Executes hybrid retrieval (Dense Vector + BM25 Sparse + Reranker)
        via the Backend API or fallback domain database.
        """
        if not self.use_mock:
            results = await self._fetch_live_evidence(
                query, vehicle_model, subsystem, top_k
            )
            logger.info(f"Retrieved {len(results)} items from live Backend API")
            return results

        # Note
        q_lower = query.lower()
        scored_results = []

        for item in SAMPLE_EV_EVIDENCE_STORE:
            score = 0.5
            content_lower = item["content"].lower()
            title_lower = item["document_title"].lower()
            subsystem_lower = item["subsystem"].lower()

            # Note
            if subsystem and subsystem.lower() in subsystem_lower:
                score += 0.3

            # Note
            keywords = [w for w in q_lower.split() if len(w) > 3]
            match_count = sum(1 for kw in keywords if kw in content_lower or kw in title_lower)
            score += min(0.4, match_count * 0.1)

            scored_item = item.copy()
            scored_item["relevance_score"] = min(0.99, score)
            scored_results.append(scored_item)

        # Note
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_items = scored_results[:top_k]
        return [EvidenceSource(**item) for item in top_items]

    async def get_dtc_details(self, dtc_code: str) -> Optional[DTCRecord]:
        """Lookup diagnostic trouble code details."""
        code_clean = dtc_code.strip().upper().replace(" ", "")
        if not self.use_mock:
            if not self.api_key:
                raise RuntimeError(
                    "Live retrieval is enabled, but HYBRID_RETRIEVAL_API_KEY is not configured."
                )
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{self.base_url.rstrip('/')}/dtc/{code_clean}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                if response.status_code == 404:
                    return None
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Live DTC lookup returned HTTP {response.status_code}."
                    )
                return DTCRecord.model_validate(response.json())
            except RuntimeError:
                raise
            except (httpx.RequestError, ValueError, TypeError) as error:
                raise RuntimeError(
                    "Live DTC lookup failed; sample data was not used."
                ) from error

        if code_clean in SAMPLE_DTC_DATABASE:
            return DTCRecord(**SAMPLE_DTC_DATABASE[code_clean])
        if code_clean in GENERIC_DTC_DATABASE:
            return DTCRecord(**GENERIC_DTC_DATABASE[code_clean])
        return None

    async def get_component_specifications(self, component_name: str) -> Optional[ComponentSpec]:
        """Lookup component pinouts and operating thresholds."""
        if not self.use_mock:
            return None
        comp_key = component_name.lower().replace(" ", "_")
        for key, spec_data in SAMPLE_COMPONENT_SPECS.items():
            if key in comp_key or comp_key in key or any(w in comp_key for w in key.split("_")):
                return ComponentSpec(**spec_data)
        return None

    async def fetch_raw_chunks_from_backend(
        self,
        query: str,
        top_k: int = 5,
        vehicle_model: Optional[str] = None,
        subsystem: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generic call to retrieve N separate chunks (default 5) from the teammate's vector search / backend API.

        ========================================================================
        NOTE FOR BACKEND TEAMMATE:
        - Update endpoint URL, HTTP method (POST/GET), or request body below
          to match your backend vector search API contract.
        - By default, it expects a response containing 5 chunks in your JSON schema.
        ========================================================================
        """
        if not self.use_mock:
            sources = await self._fetch_live_evidence(
                query, vehicle_model, subsystem, top_k
            )
            chunks: List[Dict[str, Any]] = []
            for source in sources:
                metadata = source.metadata
                provenance = metadata.get("provenance")
                provenance = provenance if isinstance(provenance, dict) else {}
                chunk_metadata = metadata.get("chunk_metadata")
                chunk_metadata = (
                    dict(chunk_metadata)
                    if isinstance(chunk_metadata, dict)
                    else {}
                )
                if source.section:
                    chunk_metadata.setdefault("section", source.section)
                if source.subsystem:
                    chunk_metadata.setdefault("subsystem", source.subsystem)
                if source.page_number is not None:
                    chunk_metadata.setdefault("page", source.page_number)
                chunks.append(
                    {
                        "chunk_id": source.source_id,
                        "record_id": metadata.get("record_id"),
                        "source_document": metadata.get("source_document"),
                        "source_type": source.document_type,
                        "content_type": metadata.get(
                            "content_type", source.document_type
                        ),
                        "title": source.document_title,
                        "text": source.content,
                        "score": source.relevance_score,
                        "provenance": provenance,
                        "metadata": chunk_metadata,
                    }
                )
            return chunks

        # Note
        # Note
        # Note
        q_lower = query.lower()
        
        fallback_chunks = [
            {
                "chunk_id": "2025-450S-QSG_page_1_chunk_0",
                "record_id": "2025-450S-QSG_page_1",
                "source_document": "2025-450S-QSG",
                "source_type": "official",
                "content_type": "pdf",
                "title": "2025-450S Quick Start Guide - Home Charging Specs",
                "text": "The Ather 450S comes with backwards compatibility for existing 700W chargers. Recharging from 0-80% takes 3 hr 30 min, and 0-100% takes 4 hr 30 min using the standard home 700W unit. Standard home charging time via portable charger is 6 hr 36 min (0-80%) and 8 hr 36 min (0-100%).",
                "score": 0.8900,
                "provenance": {
                    "document_id": "2025-450S-QSG",
                    "url": "https://media.atherenergy.com/2025-450S-QSG.pdf",
                    "source_type": "official",
                    "document_type": "quick_start_guide",
                    "manufacturer": "Ather Energy",
                    "vehicle_model": "Ather 450S",
                    "retrieved_at": "2026-10-26T07:21:12.151865+00:00"
                },
                "metadata": {
                    "category": "charging-grid",
                    "group": "Ather 450 Series",
                    "page": 1
                }
            },
            {
                "chunk_id": "2025-450S-QSG_page_2_chunk_0",
                "record_id": "2025-450S-QSG_page_2",
                "source_document": "2025-450S-QSG",
                "source_type": "official",
                "content_type": "pdf",
                "title": "Ather 450 Series - FallSafe Safety System",
                "text": "FallSafe is an intelligent safety feature that cuts off power to the motor whenever a vehicle fall is detected, stopping the scooter immediately. In addition, hazard turn indicators automatically flash to alert traffic.",
                "score": 0.8450,
                "provenance": {
                    "document_id": "ather_fallsafe_spec",
                    "url": "https://www.atherenergy.com/faq/explore-buy/ather-450-series",
                    "source_type": "official",
                    "document_type": "faq",
                    "manufacturer": "Ather Energy",
                    "vehicle_model": "Ather 450 Series",
                    "retrieved_at": "2026-10-26T07:21:12.151865+00:00"
                },
                "metadata": {
                    "category": "safety",
                    "group": "Safety Controls",
                    "page": 2
                }
            },
            {
                "chunk_id": "2025-450S-QSG_page_3_chunk_0",
                "record_id": "2025-450S-QSG_page_3",
                "source_document": "2025-450S-QSG",
                "source_type": "official",
                "content_type": "pdf",
                "title": "Ather Roadside Assistance - Discharge Immobility",
                "text": "In the event that a fully discharged battery leaves your vehicle immovable on the road, Roadside Assistance (RSA) provides towing services to the nearest Ather touchpoint or grid station with no questions asked.",
                "score": 0.8120,
                "provenance": {
                    "document_id": "ather_rsa_terms",
                    "url": "https://www.atherenergy.com/faq/owning-ather/ather_roadside_assistance",
                    "source_type": "official",
                    "document_type": "faq",
                    "manufacturer": "Ather Energy",
                    "vehicle_model": "Ather 450 Series",
                    "retrieved_at": "2026-10-26T07:21:12.151865+00:00"
                },
                "metadata": {
                    "category": "owning-ather",
                    "group": "Roadside Assistance",
                    "page": 3
                }
            },
            {
                "chunk_id": "2025-450S-QSG_page_4_chunk_0",
                "record_id": "2025-450S-QSG_page_4",
                "source_document": "2025-450S-QSG",
                "source_type": "official",
                "content_type": "pdf",
                "title": "Ather Eight70 Battery Warranty & Degradation Assurance",
                "text": "The Eight70 Warranty covers the battery for 8 years or 80 000 km. It guarantees the battery health will remain above 70% state-of-health (SOH); if health drops below 70%, the battery is replaced free of charge.",
                "score": 0.7890,
                "provenance": {
                    "document_id": "ather_eight70_warranty",
                    "url": "https://www.atherenergy.com/faq/owning-ather/ather-battery-warranty",
                    "source_type": "official",
                    "document_type": "warranty",
                    "manufacturer": "Ather Energy",
                    "vehicle_model": "Ather 450 Series",
                    "retrieved_at": "2026-10-26T07:21:12.151865+00:00"
                },
                "metadata": {
                    "category": "warranty",
                    "group": "Battery Warranty",
                    "page": 4
                }
            },
            {
                "chunk_id": "2025-450S-QSG_page_5_chunk_0",
                "record_id": "2025-450S-QSG_page_5",
                "source_document": "2025-450S-QSG",
                "source_type": "official",
                "content_type": "pdf",
                "title": "Ather 450S High Voltage Precaution & Socket Grounding",
                "text": "Always plug the charger into a properly earthed 15A socket equipped with surge protection. Never operate chargers with damaged cables or in standing water. Ensure main circuit breaker is off during any electrical inspection.",
                "score": 0.7200,
                "provenance": {
                    "document_id": "2025-450S-QSG",
                    "url": "https://media.atherenergy.com/2025-450S-QSG.pdf",
                    "source_type": "official",
                    "document_type": "quick_start_guide",
                    "manufacturer": "Ather Energy",
                    "vehicle_model": "Ather 450S",
                    "retrieved_at": "2026-10-26T07:21:12.151865+00:00"
                },
                "metadata": {
                    "category": "safety",
                    "group": "General Safety",
                    "page": 5
                }
            }
        ]

        return fallback_chunks[:top_k]

backend_adapter = BackendRetrievalAdapter()
