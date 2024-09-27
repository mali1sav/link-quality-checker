import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def tfidf_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity * 100
st.title("Cosine Similarity Checker")

text1_input = st.text_area("Text 1", """Source Summary:

Mouser.co.uk is likely an online distributor for electronic components, specializing in semiconductors and other electronic components like LEDs and resistors, based in the UK.

SERP Titles:

Electronic Components Distributor - Mouser Electronics ..., Help Center, Credit Application, Contact Centre, Mouser Search Tools, Return Merchandise Authorization (RMA) Request, LEMO Connectors Distributor, Grayhill Switches Distributor | Mouser United Kingdom, Telemecanique Sensor Distributor, Mouser Service | Order with Confidence, Harwin Connectors Distributor, SparkFun Electronics Distributor | Mouser United Kingdom, Cree LEDs Distributor, Omron Electronic Components Distributor, Littelfuse Distributor, Welwyn Resistors & Components Distributor - TT Electronics, Amphenol FCI Connectors Distributor, Electronic Component Manufacturers by Category, Mouser Electronics, Inc. United Kingdom, Soracom Distributor

SERP Descriptions:

Mouser Electronics stocks the world's widest selection of semiconductors and electronic components. Automotive Connectors., Need assistance with an order? The Mouser Online Help Center can assist you in placing an order, building a project and searching for a product., For your convenience, Mouser has NET Terms options available for businesses, schools and government agencies. How does this benefit my company?, Mouser Electronics Artisan Building Hillbottom Road High Wycombe Buckinghamshire HP12 4HJ United Kingdom Office Hours (MF): 8:30am to 5pm GMT, Mouser Search Tools offer a variety of options for searching on Mouser.com including Translated Search, Type Ahead Search and more., Complete the form below in order to request a return authorisation for products purchased from Mouser Electronics. MyMouser account holders can log in and ..., LEMO distributor Mouser Electronics stocks LEMO connectors which are known for exceptional quality and reliability., Grayhill is vertically integrated, allowing the company to use in-house molding, tooling, testing, and manufacturing to bring custom products to market faster., XT Capacitive Proximity Sensors. Provides non-contact sensing of any material up to 20mm, regardless of material or conductivity., Mouser has delivered a 98.6% satisfaction rating based on thousands of customer surveys. Our commitment to providing excellent customer service, order accuracy ..., Harwin distributor Mouser Electronics stocks Harwin connectors including cable assemblies, vertical connectors, Datamate connectors, and terminals., SparkFun Distributor Mouser Electronics now carries SparkFun Electronics., Cree distributor Mouser Electronics provides Cree LED Lighting Solutions and products including Cree LEDs and Power products., Omron Electronic Components is the Americas subsidiary of Omron Corporation, a $7 billion global leading supplier of electronics and control system components ..., Littelfuse distributor Mouser Electronics stocks Littelfuse products including Fuses, Varistors, TVS Diodes, & more., Welwyn Components distributor Mouser Electronics stocks many Welwyn resistors and components which are available for immediate purchase., Amphenol FCI connector distributor Mouser Electronics stocks FCI connectors including DIN, D-SUB, GIG-Array, Quickie and many more., All Capacitors, Circuit Protection, Computing Connectors, Diodes & Rectifiers, Embedded Processors & Controllers, Embedded Solutions, EMI/RFI Components, ..., Electronic Component Parts Distributor, order online, no minimum order. Semiconductors, Connectors, Embedded, Optoelectronics, Capacitors, Resistors, ..., Soracom is a cellular network provider built for IoT devices and reliable connectivity across urban and rural areas around the world. Experience multi-carrier …""")
text2_input = st.text_area("Text 2", """S Online, or RS Components, is a global distributor of industrial and electronic products, serving a broad range of sectors and industries with a comprehensive selection of tools, equipment, and components.

Key Target Sectors:
Manufacturing and Industrial: Supports companies in automotive, aerospace, electronics, and general manufacturing sectors, providing components for automation, control, and maintenance systems.
Engineering and R&D: Targets engineers and researchers in electronics, mechanical engineering, and product design, offering prototyping tools, test & measurement equipment, and custom automation solutions.
Healthcare and Pharmaceuticals: Supplies equipment for medical devices, diagnostic tools, and pharmaceutical manufacturing, ensuring high standards of safety and automation.
Energy and Utilities: Provides solutions for renewable energy, electrical distribution, and utility maintenance, focusing on electrification, cabling, and automation for optimal performance and safety.
Construction and Facilities Management: Supports infrastructure projects, offering products for electrical installations, HVAC, facility maintenance, and essential safety gear.
Product Ranges:
RS offers an extensive portfolio of products to meet the specific needs of professionals in these industries:

Connectors: A wide selection of connectors for power, data, and signal transmission, suitable for industries like manufacturing, automotive, and healthcare.
Electrification: Electrical equipment such as circuit protection, relays, and contactors for energy, utilities, and industrial automation.
Cables & Wires: High-quality cables and wires for power transmission, data connectivity, and industrial automation across a variety of sectors.
Test & Measurement Instruments: Multimeters, oscilloscopes, thermal cameras, and other instruments for testing and diagnosing electrical and mechanical systems.
Automation & Control: Programmable Logic Controllers (PLCs), motors, drives, sensors, and robotic systems for enhancing industrial automation and control processes.
Safety: Personal Protective Equipment (PPE), safety signage, lockout-tagout systems, and safety switches designed to protect workers in hazardous environments.
Mechanical & Fluid Power: Bearings, fasteners, seals, and hydraulic systems for managing power transmission and fluid control in manufacturing and engineering applications.
Facilities & Maintenance: HVAC systems, lighting, janitorial supplies, and facility management tools to ensure the smooth operation and upkeep of large industrial and public facilities.
Semiconductors: Although not a core focus, RS provides a range of semiconductors such as microcontrollers and transistors to support specialised electronic projects.
RS also offers technical support, design services, and procurement solutions, making them a go-to partner for engineers and businesses looking for reliable components and services in the industrial and technical space..""")

if st.button("Calculate Similarity"):
    text1 = preprocess(text1_input)
    text2 = preprocess(text2_input)
    similarity = tfidf_cosine_similarity(text1, text2)
    st.write(f"Similarity: {similarity:.2f}%")

