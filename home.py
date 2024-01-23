import streamlit as st
import gensim
from gensim import corpora, models
import re
import heapq
import whisper
import os
from pytube import YouTube
from pathlib import Path
import shutil
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("wordnet")

#### topic labeling ##########

insurance_keywords = [
    "actuary", "claims", "coverage", "deductible", "policyholder", "premium", "underwriter", "risk",
    "assessment", "insurable", "interest", "loss", "ratio", "reinsurance", "actuarial", "tables",
    "property", "damage", "liability", "flood", "insurance", "term", "life", "whole", "health",
    "auto", "homeowners", "marine", "crop", "catastrophe", "umbrella", "pet", "travel", "professional",
    "disability", "long-term", "care", "annuity", "pension", "plan", "group", "insurtech", "insured",
    "insurer", "subrogation", "adjuster", "third-party", "administrator", "excess", "and", "surplus",
    "lines", "captives", "workers", "compensation", "fraud", "health", "savings", "account",
    "maintenance", "organization", "preferred", "provider", "property", "casualty", "management",
    "broker", "healthcare", "term", "whole", "death", "benefit", "payout", "investment", "fund",
    "index", "variable", "surrender", "charge", "insurance", "broker", "healthcare", "coverage",
    "term", "life", "whole", "premium", "deductible", "copayment", "coinsurance", "exclusion",
    "pre-existing", "condition", "policy", "limit", "beneficiary", "cash", "value", "dividend",
    "annuity", "term", "whole", "premium", "death", "benefit", "payout", "investment", "fund",
    "index", "variable", "surrender", "charge", "insurance", "broker", "healthcare", "coverage",
    "term", "life", "whole", "premium", "deductible", "copayment", "coinsurance", "exclusion",
    "pre-existing", "condition", "policy", "limit", "beneficiary", "cash", "value", "dividend",
    "annuity", "term", "whole", "premium", "death", "benefit", "payout", "investment", "fund",
    "index", "variable", "surrender", "charge", "insurance", "broker", "healthcare", "coverage",
    "term", "life", "whole", "premium", "deductible", "copayment", "coinsurance", "exclusion",
    "pre-existing", "condition", "policy", "limit", "beneficiary", "cash", "value", "dividend",
    "annuity", "term", "whole", "premium", "death", "benefit", "payout", "investment", "fund",
    "index", "variable", "surrender", "charge"
]



finance_keywords = [
    "asset", "liability", "equity", "capital", "portfolio", "dividend", "financial statement", "balance sheet", "income statement", "cash flow statement",
    "retained earnings", "ratio", "valuation", "bond", "stock", "mutual fund", "exchange-traded fund", "hedge fund", "private equity", "venture capital", 
    "mergers", "acquisitions", "initial public offering", "secondary market", "primary market", "securities", "derivative", "option", "futures", 
    "forward contract", "swaps", "commodities", "credit rating", "score", "report", "bureau", "history", "limit", "utilization", "counseling", 
    "credit card", "debit card", "ATM", "bankruptcy", "foreclosure", "debt consolidation", "taxes", "tax return", "deduction", "credit", "bracket", 
    "taxable income", "investment", "gain", "loss", "dividend yield", "return", "risk", "interest rate", "inflation", "deflation", "monetary policy", 
    "fiscal policy", "quantitative easing", "currency exchange", "foreign exchange", "exchange rate", "treasury bond", "savings account", "loan", 
    "mortgage", "interest-only loan", "financial planning", "wealth management", "retirement planning", "401(k)", "IRA", "annuity", "pension plan", 
    "budgeting", "savings", "investment strategy", "advisor", "literacy", "education", "compound interest", "time", "value", "money", "capital budgeting", 
    "modeling", "cash management", "liquidity", "market risk", "systemic risk", "credit risk", "operational risk", "liquidity risk", "market liquidity", 
    "hedging", "derivatives", "institution", "bank", "credit union", "investment bank", "central bank", "Federal Reserve", "stock exchange", "manager", 
    "portfolio manager", "analyst",
]



banking_capital_markets_keywords = [
    "bank", "credit union", "savings and loan association", "commercial bank", "investment bank", "retail bank", "wholesale bank", "online bank", "mobile banking",
    "checking account", "savings account", "money market account", "certificate of deposit", "loan", "mortgage", "home equity loan", "line of credit", "credit card",
    "debit card", "ATM", "automated clearing house", "wire transfer", "ACH", "SWIFT", "international banking", "foreign exchange", "forex", "currency exchange",
    "central bank", "Federal Reserve", "interest rate", "inflation", "deflation", "monetary policy", "fiscal policy", "quantitative easing", "securities", "stock",
    "bond", "mutual fund", "exchange-traded fund", "hedge fund", "private equity", "venture capital", "investment management", "portfolio management", "wealth management",
    "financial planning", "asset management", "investment portfolio", "diversification", "equity market", "stock exchange", "bull market", "bear market",
    "market capitalization", "dividend yield", "stock index", "market trend", "market volatility", "stockbroker", "trading floor", "market order", "limit order",
    "stop order", "day trading", "margin trading", "leverage", "financial instrument", "derivatives", "options", "futures contract", "commodities trading",
    "trading strategy", "risk management", "portfolio risk", "financial regulation", "banking regulation", "securities regulation", "financial institution",
    "financial market", "credit risk", "liquidity risk", "operational risk", "market risk", "systemic risk", "financial crisis", "credit rating agency",
    "credit default swap", "mortgage-backed securities", "securitization", "financial innovation", "financial technology", "fintech", "blockchain", "cryptocurrency",
    "digital banking", "robo-advisor", "mobile payments", "contactless payments", "financial inclusion", "green finance", "sustainable investing", "ethical investing",
    "socially responsible investing", "financial education", "financial literacy",
]



healthcare_life_sciences_keywords = [
    "medical device", "pharmaceutical", "biotechnology", "clinical trial", "FDA", "healthcare provider", "healthcare plan", "healthcare insurance", "patient", "doctor",
    "nurse", "pharmacist", "hospital", "clinic", "healthcare system", "healthcare policy", "public health", "healthcare IT", "electronic health record", "telemedicine",
    "personalized medicine", "genomics", "proteomics", "clinical research", "drug development", "drug discovery", "medicine", "health", "wellness", "biomedical", "biomedicine",
    "pathology", "pharmacology", "vaccine", "immunology", "virology", "microbiology", "epidemiology", "biostatistics", "healthcare management", "healthcare administration",
    "health informatics", "medical imaging", "radiology", "nuclear medicine", "neuroscience", "cardiology", "oncology", "dermatology", "gastroenterology", "ophthalmology",
    "dentistry", "orthopedics", "physical therapy", "occupational therapy", "speech therapy", "nutrition", "dietetics", "pharmacy", "pharmacogenomics", "healthcare ethics",
    "bioethics", "medical research", "scientific", "laboratory", "clinical laboratory", "medical technology", "medical equipment", "medical coding", "healthcare finance",
    "health economics", "health insurance", "healthcare reform", "telehealth", "eHealth", "mHealth", "global health", "tropical medicine", "infectious diseases", "chronic diseases",
    "rare diseases", "mental health", "psychiatry", "psychology", "psychotherapy", "rehabilitation", "palliative care", "geriatrics", "pediatrics", "obstetrics", "gynecology",
    "family medicine", "preventive medicine", "vital signs", "diagnosis", "treatment", "therapy", "patient care", "clinical pathway", "health assessment", "patient education",
    "health promotion", "prevention", "screening", "vaccination", "hygiene", "sanitation", "vector control", "bioterrorism", "natural disaster", "emergency response",
    "community health", "primary care", "secondary care", "tertiary care", "health disparities", "health equity", "pharmaceutical industry", "biotech industry",
    "medical device industry", "healthcare innovation", "regulatory affairs", "clinical trials", "evidence-based medicine", "medical ethics", "clinical ethics",
    "research ethics", "patient rights", "confidentiality", "health information privacy", "medical law", "health law", "biomedical engineering", "bioinformatics",
    "health data", "health communication", "patient engagement", "shared decision making", "medical humanities", "medical education", "medical training",
    "professional development", "continuing medical education", "medical conferences", "academic medicine", "medical journalism", "scientific communication",
    "healthcare marketing", "healthcare branding", "healthcare advertising", "healthcare technology", "digital health", "wearable technology", "health apps",
    "medical imaging technology", "genetic testing", "personalized healthcare", "precision medicine", "big data in healthcare", "data analytics in healthcare",
    "artificial intelligence in healthcare", "machine learning in healthcare", "healthcare cybersecurity", "healthcare information systems", "healthcare interoperability",
    "patient records", "electronic medical records", "healthcare standards", "clinical coding", "healthcare quality", "patient safety", "medical error", "healthcare accreditation",
    "healthcare governance", "healthcare compliance", "healthcare ethics committee", "healthcare risk management", "healthcare audit", "healthcare consulting",
    "health economics", "health policy", "healthcare regulation", "healthcare law", "healthcare advocacy", "patient advocacy", "health insurance", "healthcare financing",
    "healthcare economics", "healthcare delivery", "primary care delivery", "hospital care", "outpatient care", "home healthcare", "urgent care", "ambulatory care",
    "integrated care", "coordinated care", "healthcare workforce", "physician", "nurse practitioner", "physician assistant", "medical assistant", "healthcare administrator",
    "healthcare manager", "healthcare executive", "health informatician", "clinical informatician", "healthcare educator", "healthcare researcher", "clinical researcher",
    "health economist", "health policy analyst", "healthcare consultant", "healthcare IT specialist", "medical writer", "healthcare journalist", "healthcare photographer",
    "medical illustrator", "healthcare artist", "healthcare translator", "medical interpreter", "healthcare librarian", "medical librarian", "healthcare ethicist",
    "clinical ethicist", "healthcare lawyer",
]



law_keywords = [
    "law", "legal", "attorney", "lawyer", "litigation", "arbitration", "dispute resolution", "contract law", "intellectual property", "corporate law", "labor law",
    "tax law", "real estate law", "environmental law", "criminal law", "family law", "immigration law", "bankruptcy law", "lawsuit", "legal system", "court", "judge",
    "jury", "legal rights", "legal obligations", "legal proceedings", "legal counsel", "legal code", "legal precedent", "legal research", "legal ethics", "legal practice",
    "legal theory", "legal profession", "legal advice", "legal document", "legal dispute", "legal representation", "legal procedure", "legal framework", "legal terminology",
    "legal terminology", "civil law", "common law", "constitutional law", "international law", "human rights law",
]



sports_keywords = [
    "sports", "football", "basketball", "baseball", "hockey", "soccer", "golf", "tennis", "olympics", "athletics", "coaching", "sports management",
    "sports medicine", "sports psychology", "sports broadcasting", "sports journalism", "esports", "fitness", "world cup", "olympic", "final", "athletes",
    "competition", "championship", "team", "league", "score", "goal", "match", "tournament", "athlete", "coach", "manager", "umpire", "referee", "fans",
    "spectators", "stadium", "arena", "field", "court", "pitch", "course", "ring", "track", "training", "exercise", "workout", "physical", "conditioning",
    "strength", "endurance", "speed", "agility", "flexibility", "teamwork", "strategy", "tactics", "offense", "defense", "goalkeeper", "striker", "midfielder",
    "defender", "forward", "center", "quarterback", "pitcher", "hitter", "goalie", "swimmer", "runner", "cyclist", "climber", "skater", "surfer", "skier",
    "snowboarder", "athlete", "competitive", "victory", "defeat", "winning", "losing", "medal", "podium", "record", "champion", "title", "sportsmanship",
    "fair play", "fanatic", "cheering", "adrenaline", "intensity", "sports gear", "uniform", "equipment", "ball", "racket", "bat", "club", "stick", "glove",
    "helmet", "jersey", "shoes", "training camp", "scouting", "recruitment", "draft", "retirement", "hall of fame", "legends", "icon", "hero", "role model",
    "inspiration", "sports science", "injury", "rehabilitation", "performance", "achievement", "celebration", "victory parade", "sportsmanship", "team spirit",
    "rivalry", "competition", "sports venue", "sports equipment", "fans", "spectators", "cheerleaders", "halftime", "locker room", "championship", "trophy",
    "medal", "awards", "world record", "national team", "captain", "team captain", "referee", "judges", "scoring", "timekeeping", "penalty", "overtime",
    "sportsmanship", "dedication", "perseverance", "discipline", "training regimen", "nutrition", "hydration", "pre-game rituals", "post-game rituals",
    "adversary", "opponent", "friendly match", "sports analysis", "commentary", "sports broadcasting", "sports journalism", "retired athlete",
    "sports administration", "sports marketing", "endorsements", "sponsorship", "fanbase", "team colors", "mascot", "sports injuries", "recovery",
    "sports psychology", "performance-enhancing drugs", "doping", "fair play", "team dynamics", "individual sports", "team sports", "spectator sports",
    "extreme sports", "water sports", "winter sports", "summer sports", "paralympics", "adaptive sports", "sports culture", "sportsmanship", "training camp",
    "coaching staff", "sportsmanship", "training facility", "sports facility", "athletic achievement", "sports culture", "sportsmanship", "training camp",
    "coaching staff", "sportsmanship", "training facility", "sports facility", "athletic achievement",
]



media_keywords = [
    "media", "entertainment", "film", "television", "radio", "music", "news", "journalism", "publishing", "public relations", "advertising", "marketing",
    "social media", "digital media", "animation", "graphic design", "web design", "video production", "broadcasting", "content creation", "storytelling",
    "cinematography", "scriptwriting", "screenwriting", "editing", "post-production", "cinema", "TV series", "documentary", "media coverage", "press",
    "media industry", "media outlets", "media literacy", "media studies", "media ethics", "media influence", "media bias", "media consumption",
    "media regulation", "media landscape", "media convergence", "media production", "media technology", "media trends", "streaming", "podcasting",
    "vlogging", "influencers", "celebrity culture", "pop culture", "filmography", "television programs", "radio broadcasting", "music genres", "music production",
    "music festivals", "journalistic ethics", "editorial standards", "media censorship", "advertising agencies", "branding", "market research", "publicity",
    "viral marketing", "social media platforms", "digital marketing", "content marketing", "SEO", "SEM", "web development", "UX design", "UI design",
    "user engagement", "online presence", "animated films", "animated series", "character design", "motion graphics", "film festivals", "screenplay",
    "documentary filmmaking", "photojournalism", "news reporting", "broadcast journalism", "media communication", "media psychology", "media impact",
    "media ownership", "media literacy", "media convergence", "media production", "media technology", "media trends", "streaming", "podcasting", "vlogging",
    "influencers", "celebrity culture", "pop culture", "filmography", "television programs", "radio broadcasting", "music genres", "music production",
    "music festivals", "journalistic ethics", "editorial standards", "media censorship", "advertising agencies", "branding", "market research", "publicity",
    "viral marketing", "social media platforms", "digital marketing", "content marketing", "SEO", "SEM", "web development", "UX design", "UI design",
    "user engagement", "online presence", "animated films", "animated series", "character design", "motion graphics", "film festivals", "screenplay",
    "documentary filmmaking", "photojournalism", "news reporting", "broadcast journalism", "media communication", "media psychology", "media impact",
    "media ownership", "media literacy", "media convergence", "media production", "media technology", "media trends", "streaming", "podcasting", "vlogging",
    "influencers", "celebrity culture", "pop culture", "filmography", "television programs", "radio broadcasting", "music genres", "music production",
    "music festivals", "journalistic ethics", "editorial standards", "media censorship", "advertising agencies", "branding", "market research", "publicity",
    "viral marketing", "social media platforms", "digital marketing", "content marketing", "SEO", "SEM", "web development", "UX design", "UI design",
    "user engagement", "online presence", "animated films", "animated series", "character design", "motion graphics", "film festivals", "screenplay",
    "documentary filmmaking", "photojournalism", "news reporting", "broadcast journalism", "media communication", "media psychology", "media impact",
    "media ownership", "media literacy", "media convergence", "media production", "media technology", "media trends", "streaming", "podcasting", "vlogging",
    "influencers", "celebrity culture", "pop culture", "filmography", "television programs", "radio broadcasting", "music genres", "music production",
]



manufacturing_keywords = [
    "manufacturing", "production", "assembly", "logistics", "supply chain", "quality control", "lean manufacturing", "six sigma", "industrial engineering",
    "process improvement", "machinery", "automation", "aerospace", "automotive", "chemicals", "construction materials", "consumer goods", "electronics",
    "semiconductors", "factory", "plant", "facility", "workshop", "warehouse", "inventory", "raw materials", "finished goods", "efficiency", "productivity",
    "efficiency", "output", "workforce", "labor", "management", "operations", "manufacturing process", "technology", "innovation", "advanced manufacturing",
    "smart manufacturing", "digitalization", "IoT in manufacturing", "robotics", "3D printing", "additive manufacturing", "CAD", "CAM", "CNC machining", "molding",
    "casting", "welding", "fabrication", "tooling", "maintenance", "quality assurance", "ISO standards", "sustainability", "environmental impact",
    "green manufacturing", "renewable energy", "supply chain management", "just-in-time", "kanban", "total quality management", "continuous improvement",
    "global manufacturing", "offshoring", "reshoring", "outsourcing", "vendor", "partnership", "collaboration", "manufacturing trends", "industry 4.0",
    "smart factories", "digital twin", "data analytics in manufacturing", "sensors", "connectivity", "big data", "cloud computing", "ERP",
    "enterprise resource planning", "SCADA", "process control", "automation technologies", "workplace safety", "occupational health", "regulatory compliance",
    "material handling", "logistics management", "packaging", "shipping", "distribution", "global supply chain", "logistics network", "industrial design",
    "product design", "prototyping", "innovative materials", "biotechnology in manufacturing", "nanotechnology", "advanced materials",
]



automotive_keywords = [
    "automotive", "cars", "trucks", "SUVs", "electric vehicles", "hybrid vehicles", "autonomous vehicles", "car manufacturing", "automotive design", "car dealerships",
    "auto parts", "vehicle maintenance", "car rental", "fleet management", "telematics", "automobile", "engine", "transmission", "fuel efficiency", "vehicle safety",
    "collision avoidance", "autonomous driving", "self-driving", "car insurance", "fuel economy", "exhaust system", "emission standards", "catalytic converter",
    "hybrid technology", "electric car charging", "alternative fuels", "battery technology", "tire technology", "suspension system", "brake system",
    "automotive engineering", "manufacturing process", "supply chain", "quality control", "assembly line", "automotive industry", "car market",
    "electric vehicle market", "connected cars", "infotainment system", "navigation system", "driver assistance", "parking assistance", "adaptive cruise control",
    "lane departure warning", "traffic sign recognition", "airbags", "crash test", "safety ratings", "automotive innovation", "concept cars", "luxury cars",
    "sports cars", "SUV market", "truck market", "car interior", "car exterior", "car customization", "car accessories", "auto financing", "car loan", "leasing",
    "used cars", "car resale value", "automotive technology", "smart cars", "IoT in automotive", "vehicle-to-vehicle communication",
    "vehicle-to-infrastructure communication", "car hacking", "autonomous taxis", "ride-sharing", "electric charging infrastructure", "charging stations",
    "automotive startups", "automotive trends", "future of mobility", "zero-emission vehicles", "automotive maintenance", "oil change", "tire rotation",
    "brake service", "engine diagnostics", "car wash", "fluid levels check", "auto detailing", "vehicle inspection", "roadside assistance", "towing service",
]



telecom_keywords = [
    "telecom", "telecommunications", "wireless", "networks", "internet", "broadband", "fiber optics", "5G", "telecom infrastructure", "telecom equipment", "VoIP",
    "satellite communications", "mobile devices", "smartphones", "telecom services", "telecom regulation", "telecom policy", "telephony", "communication", "data transfer",
    "data transmission", "ISP", "internet service provider", "router", "modem", "Wi-Fi", "cellular network", "LTE", "3G", "4G", "network security", "firewall", "encryption",
    "protocol", "IP address", "DNS", "ISP", "Internet", "Wi-Fi hotspot", "bandwidth", "latency", "ping", "cloud computing", "data center", "telecom tower", "antenna",
    "radio waves", "spectrum", "DSL", "cable modem", "satellite internet", "communication protocol", "packet", "data packet", "routing", "switching", "TCP/IP", "OSI model",
    "telecom engineer", "telecom technician", "telecom infrastructure", "ISP", "internet backbone", "telecom standards", "digital divide", "last mile", "access network",
    "core network", "wireless technology", "telecom industry", "ITU", "FCC", "regulatory compliance", "net neutrality", "telecom market", "telecom company", "telecom innovation",
    "mobile communication", "landline", "VoLTE", "OTT", "fiber optic cable", "dark fiber", "broadband speed", "ISP speed", "internet outage", "telecom policy",
    "telecom investment", "telecom trends", "IoT connectivity", "5G technology", "telecom equipment", "telecom network", "telecom satellite", "telecom research",
    "internet privacy", "digital communication", "telecom solutions", "telecom infrastructure development",
]



information_technology_keywords = [
    "Artificial intelligence", "Machine learning", "Data Science", "Big Data", "Cloud Computing", "Cybersecurity", "Information security", "Network security",
    "Blockchain", "Cryptocurrency", "Internet of things", "IoT", "Web development", "Mobile development", "Frontend development", "Backend development",
    "Software engineering", "Software development", "Programming", "Database", "Data analytics", "Business intelligence", "DevOps", "Agile", "Scrum",
    "Product management", "Project management", "IT consulting", "IT service management", "ERP", "CRM", "SaaS", "PaaS", "IaaS", "Virtualization",
    "Artificial reality", "AR", "Virtual reality", "VR", "Gaming", "E-commerce", "Digital marketing", "SEO", "SEM", "Content marketing", "Social media marketing",
    "User experience", "UX design", "UI design", "Cloud-native", "Microservices", "Serverless", "Containerization", "API", "Algorithm", "Automation", "Data mining",
    "Deep learning", "Neural network", "Quantum computing", "Edge computing", "Fintech", "Insurtech", "Healthtech", "Edtech", "Telecom", "Wireless", "Networks", "Internet",
    "Broadband", "Fiber optics", "5G", "Telecom infrastructure", "Telecom equipment", "VoIP", "Satellite communications", "Mobile devices", "Smartphones", "Telecom services",
    "Telecom regulation", "Telecom policy", "Open source", "Linux", "Operating system", "Programming language", "Code", "Debugging", "Software testing", "User interface",
    "User interaction", "Accessibility", "Wireframe", "Prototype", "Data center", "Server", "Cluster", "Backup", "Disaster recovery", "Virtual machine", "Network architecture",
    "Firewall", "Encryption", "Authentication", "Authorization", "Cyber attack", "Malware", "Phishing", "Incident response", "Computer network", "LAN", "WAN", "Wi-Fi",
    "Bluetooth", "Routing", "Switching", "Endpoint security", "Penetration testing", "IT governance", "IT strategy", "Digital transformation", "IT infrastructure",
    "Cloud security", "Privacy", "Compliance", "IT risk management", "IT audit", "Data privacy", "Data protection", "Information governance", "IT policy", "Data breach",
    "Security awareness", "IT architecture", "IT roadmap", "IT budget", "IT leadership", "IT collaboration", "IT innovation", "IT trends", "IT standards", "IT certification",
]



politics_keywords = [
    "government", "election", "democracy", "polity", "legislation", "policy", "law", "congress", "senate", "parliament",
    "president", "prime minister", "campaign", "voting", "political party", "politician", "diplomacy", "international relations",
    "foreign policy", "constituency", "constituent", "citizenship", "civil rights", "human rights", "freedom", "democratic process",
    "public office", "public service", "public policy", "governance", "bureaucracy", "lobbying", "activism", "protest", "dissent",
    "reform", "corruption", "transparency", "accountability", "separation of powers", "checks and balances", "political ideology",
    "nationalism", "patriotism", "partisanship", "bipartisanship", "coalition", "cabinet", "opposition", "republic", "monarchy",
    "authoritarianism", "totalitarianism", "anarchy", "socialism", "capitalism", "fascism", "communism", "liberalism", "conservatism",
    "libertarianism", "populism", "revolution", "civil unrest", "war", "diplomat", "negotiation", "treaty", "security", "defense",
    "terrorism", "intelligence", "surveillance", "conflict", "foreign aid", "embargo", "sanctions", "constitution", "amendment",
    "secession", "judiciary", "judicial system", "ruling", "lawmaker", "legislative branch", "executive branch", "judicial branch",
    "civic duty", "civic engagement", "campaign finance", "lobbyist", "political science", "public opinion", "media bias",
    "political commentator", "spin doctor", "policy analysis", "political theory", "political philosophy", "electioneering",
    "political debate", "polling", "political discourse", "grassroots movement", "political activism", "coup", "impeachment",
    "redistricting", "electoral college", "gerrymandering", "ballot", "referendum", "initiative", "recall", "voter turnout",
    "political campaign", "political platform", "public office", "inauguration", "statecraft", "vote of no confidence",
    "political asylum", "humanitarian intervention", "political instability", "regime change", "state sovereignty",
    "power sharing", "conflict resolution", "election fraud", "political correctness", "political satire", "political rally",
]


education_keywords = [
    "education", "learning", "teaching", "school", "classroom", "students", "teachers", "curriculum", "lesson", "lecture",
    "study", "knowledge", "skills", "academic", "degree", "certificate", "training", "instructor", "professor", "tutor",
    "pedagogy", "andragogy", "e-learning", "online education", "distance learning", "classmates", "classmate", "classwork", "homework",
    "assignment", "examination", "test", "quiz", "grade", "grading", "assessment", "evaluation", "textbook", "course",
    "major", "minor", "degree program", "research", "thesis", "dissertation", "scholarship", "grant", "student loans", "tuition",
    "fees", "admissions", "enrollment", "academic calendar", "semester", "trimester", "quarter", "graduate", "undergraduate", "postgraduate",
    "college", "university", "school district", "preschool", "elementary", "middle school", "high school", "higher education", "vocational",
    "technical school", "librarian", "library", "educational technology", "classroom technology", "smartboard", "e-book", "lecture notes", "educational resources",
    "teaching methods", "pedagogical approaches", "learning styles", "critical thinking", "problem-solving", "communication skills", "collaboration", "creativity", "motivation",
    "discipline", "class management", "educational psychology", "counseling", "academic achievement", "extracurricular activities", "student organizations", "parent-teacher conferences", "school counselor",
    "education system", "public education", "private education", "charter school", "homeschooling", "standardized testing", "educational policy", "education reform", "inclusive education", "special education",
    "learning disabilities", "gifted education", "adult education", "lifelong learning", "continuing education", "professional development", "teacher certification", "educational research", "educational administration", "educational leadership",
]


weather_keywords = [
    "weather", "forecast", "meteorology", "temperature", "humidity", "precipitation", "rainfall", 
    "snowfall", "wind speed", "wind direction", "atmosphere", "barometer", "climate", "climatology", 
    "cloud cover", "sunshine", "storm", "thunderstorm", "lightning", "hurricane", "tornado", 
    "typhoon", "monsoon", "heatwave", "cold front", "warm front", "low pressure", "high pressure", 
    "drought", "flood", "blizzard", "fog", "mist", "drizzle", "sleet", "hail", "weather map", 
    "radar", "satellite", "weather station", "climate change", "global warming", "El Niño", 
    "La Niña", "weather phenomenon", "nor'easter", "avalanche", "wind chill", "UV index", 
    "pollution", "air quality", "season", "spring", "summer", "autumn", "fall", "winter", 
    "celsius", "fahrenheit", "weather advisory", "meteorologist", "weatherman", "weatherwoman", 
    "forecasting", "weather condition", "sky", "sunrise", "sunset", "clouds", "rainbow", 
    "umbrella", "barometric pressure", "dew point", "visibility", "windbreaker", "gale", "breeze", 
    "gust", "monsoonal flow", "frontal boundary", "jet stream", "skyline", "cumulus", "stratus", 
    "cirrus", "nimbostratus", "altostratus", "cumulonimbus", "stratocumulus", "cirrostratus", 
    "cirrocumulus", "anemometer", "hygrometer", "thermometer", "weather balloon", 
    "weather satellite", "weather radar", "weather vane", "doppler radar"
]


industries = {
    "Insurance": insurance_keywords,
    "Finance": finance_keywords,
    "Banking": banking_capital_markets_keywords,
    "Healthcare": healthcare_life_sciences_keywords,
    "Legal": law_keywords,
    "Sports": sports_keywords,
    "Media": media_keywords,
    "Manufacturing": manufacturing_keywords,
    "Automotive": automotive_keywords,
    "Telecom": telecom_keywords,
    "IT": information_technology_keywords,
    "Politics": politics_keywords,
    "Education": education_keywords,
    "Weather": weather_keywords,
}


def get_root_words(word_list):
    lemmatizer = WordNetLemmatizer()
    root_words = [lemmatizer.lemmatize(word) for word in word_list]
    return root_words


def label_topic(text):
    # Count the number of occurrences of each keyword in the text for each industry
    counts = {}
    for industry, keywords in industries.items():
        count = sum(
            [
                1
                for keyword in keywords
                if re.search(r"\b{}\b".format(keyword), text, re.IGNORECASE)
            ]
        )
        counts[industry] = count

    print(counts)
    # Get the top two industries based on their counts
    # top_industries = heapq.nlargest(2, counts, key=counts.get)
    top_industries = {}
    for ind, cnt in counts.items():
        if cnt == 0:
            continue
        else:
            top_industries[ind] = cnt

    top_industries = dict(
        sorted(top_industries.items(), key=lambda item: item[1], reverse=True)
    )
    print(top_industries)

    # If only one industry was found, return it
    if len(top_industries) == 0:
        return "No maches industry on the dataset."
    # If two industries were found, return them both
    else:
        return top_industries


def perform_topic_modeling(transcript_text, num_topics=5, num_words=10):
    preprocessed_text = preprocess_text(transcript_text)

    # Create a dictionary of all unique words in the transcripts
    dictionary = corpora.Dictionary(preprocessed_text)

    # Convert the preprocessed transcripts into a bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]
    print(corpus)
    print("\n")
    # Train an LDA model with the specified number of topics
    lda_model = models.LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics
    )

    # Extract the most probable words for each topic
    topics = []
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        # Extract the top words for each topic and store in a list
        topic_words = [
            word.split("*")[1].replace('"', "").strip() for word in topic.split("+")
        ]
        topics.append((f"Topic {idx}", topic_words))

    return topics


def preprocess_text(text):
    # This example simply tokenizes the text and removes stop words
    tokens = gensim.utils.simple_preprocess(text)
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    preprocessed_text = [[token for token in tokens if token not in stop_words]]

    return preprocessed_text


@st.cache_resource
def load_model():
    model = whisper.load_model("base")
    return model


# def save_video(url, video_filename):
#     youtubeObject = YouTube(url)
#     youtubeObject = youtubeObject.streams.get_highest_resolution()
#     try:
#         youtubeObject.download()
#     except:
#         print("An error has occurred")
#     print("Download is completed successfully")

#     return video_filename


# def save_audio(url):
#     yt = YouTube(url)
#     video = yt.streams.filter(only_audio=True).first()
#     out_file = video.download()
#     base, ext = os.path.splitext(out_file)
#     file_name = base + ".mp3"
#     try:
#         os.rename(out_file, file_name)
#     except WindowsError:
#         os.remove(file_name)
#         os.rename(out_file, file_name)
#     audio_filename = Path(file_name).stem + ".mp3"
#     video_filename = save_video(url, Path(file_name).stem + ".mp4")
#     print(yt.title + " Has been successfully downloaded")
#     return yt.title, audio_filename, video_filename


# def audio_to_transcript(audio_file):
#     model = load_model()
#     result = model.transcribe(audio_file)
#     transcript = result["text"]
#     return transcript


st.set_page_config(layout="wide")

choice = st.sidebar.selectbox("Select your choice", ["On Text", "On Video", "On CSV"])


if choice == "On Text":
    st.subheader("Topic Modeling and Labeling on Text")

    # Create a text area widget to allow users to paste transcripts
    text_input = st.text_area("Paste enter text below", height=400)

    if text_input is not None:
        if st.button("Analyze Text"):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.info("Text is below")
                st.success(text_input)
            with col2:
                # Perform topic modeling on the transcript text
                topics = perform_topic_modeling(text_input)

                # Display the resulting topics in the app
                st.info("Topics in the Text")
                for topic in topics:
                    st.success(f"{topic[0]}: {', '.join(topic[1])}")
            # print(topics)
            topic_txt = ""
            for tpc in topics:
                topic_txt += " ".join(get_root_words(tpc[1]))
                topic_txt += " "
            print(topic_txt)
            with col3:
                st.info("Topic Labeling")
                labeling_text = topic_txt
                industry = label_topic(labeling_text)
                # print(industry)
                st.markdown("**Topic Labeling Industry Wise**")
                st.write(industry)

elif choice == "On Video":
    st.subheader("Topic Modeling and Labeling on Video")
    url = st.text_input("Enter URL of YouTube video:")
    # if upload_video is not None:
    if st.button("Analyze Video"):
        col1, col2, col3 = st.columns([1, 1, 1])
        # with col1:
        #     st.info("Video uploaded successfully")
        #     video_title, audio_filename, video_filename = save_audio(url)
        #     st.video(video_filename)
        # with col2:
        #     st.info("Transcript is below")
        #     print(audio_filename)
        #     transcript_result = audio_to_transcript(audio_filename)
        #     st.success(transcript_result)
        # with col3:
        #     st.info("Topic Modeling and Labeling")
        #     labeling_text = transcript_result
        #     industry = label_topic(labeling_text)
        #     st.markdown("**Topic Labeling Industry Wise**")
        #     st.write(industry)

elif choice == "On CSV":
    st.subheader("Topic Modeling and Labeling on CSV File")
    upload_csv = st.file_uploader("Upload your CSV file", type=["csv"])
    if upload_csv is not None:
        if st.button("Analyze CSV File"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.info("CSV File uploaded")
                csv_file = upload_csv.name
                with open(os.path.join(csv_file), "wb") as f:
                    f.write(upload_csv.getbuffer())
                print(csv_file)
                df = pd.read_csv(csv_file, encoding="unicode_escape")
                st.dataframe(df)
            with col2:
                data_list = df["Data"].tolist()
                industry_list = []
                for i in data_list:
                    industry = label_topic(i)
                    industry_list.append(industry)
                df["Industry"] = industry_list
                st.info("Topic Modeling and Labeling")
                st.dataframe(df)
