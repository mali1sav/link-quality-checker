import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# RS Online description
RS_ONLINE_DESCRIPTION = """
RS Online, or RS Components, is a global distributor of industrial and electronic products, serving a broad range of sectors and industries with a comprehensive selection of tools, equipment, and components.

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
RS also offers technical support, design services, and procurement solutions, making them a go-to partner for engineers and businesses looking for reliable components and services in the industrial and technical space.
"""

# Simple list of stop words
STOP_WORDS = set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with"
])

def preprocess_text(text: str) -> str:
    # Convert to lowercase and remove special characters, but keep hyphens
    text = re.sub(r'[^a-zA-Z\s\-]', ' ', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def calculate_content_similarity(source_content: str, rs_description: str) -> float:
    # Combine all text data
    all_texts = [source_content, rs_description]
    
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in all_texts]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Adjust similarity score to potentially yield higher scores
    adjusted_similarity = np.sqrt(cosine_similarities[0][0]) * 100
    
    return adjusted_similarity

st.title("Content Similarity Checker")

st.write("This tool allows you to check the similarity between your content and RS Online's description.")

# Text area for user input
user_content = st.text_area("Paste your content here (including Source Summary, SERP Titles, and SERP Descriptions):", height=300)

if st.button("Calculate Similarity"):
    if user_content:
        similarity_score = calculate_content_similarity(user_content, RS_ONLINE_DESCRIPTION)
        st.write(f"Content Similarity Score: {similarity_score:.2f}%")
        
        # Provide interpretation
        if similarity_score >= 70:
            st.success("High similarity: The content is very relevant to RS Online's industry and offerings.")
        elif similarity_score >= 50:
            st.info("Moderate similarity: The content has some relevance to RS Online's industry.")
        else:
            st.warning("Low similarity: The content may not be closely related to RS Online's industry.")
    else:
        st.error("Please enter some content before calculating similarity.")

st.write("Note: This tool uses TF-IDF vectorization and cosine similarity to calculate content similarity, focusing on industry-specific terms.")