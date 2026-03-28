# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import uuid
import traceback
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage
from sqlalchemy import create_engine, text

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Initialize Clients
genai_client = None # Initialize to None
storage_client = None
engine = None #Initialize to None

try:
    genai_client = genai.Client(api_key=API_KEY)
    storage_client = storage.Client()
    
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in environment variables.")
    
    # Increase pool size and handle disconnected sessions
    # pool_pre_ping checks if the connection is alive before using it
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
except Exception as e:
    print(f"Initialization Error: {traceback.format_exc()}")


def upload_to_gcs(file_bytes, filename):
    """Uploads a file to Google Cloud Storage and returns the public URL."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"items/{uuid.uuid4()}-{filename}")
    blob.upload_from_string(file_bytes, content_type="image/jpeg")
    return blob.public_url



@app.route('/')
def home():
    """
    Fetches companies and renders the crm dashboard template.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT company_id, industry, company_size, annual_revenue, contract_status, region 
                FROM companies 
                ORDER BY company_id DESC
                LIMIT 50
            """)
            result = conn.execute(query)
            
            companies =[]
            for row in result:
                companies.append({
                    "company_id": str(row[0]),
                    "industry": row[1],
                    "company_size": row[2],
                    "annual_revenue": row[3],
                    "contract_status": row[4],
                    "region": row[5]
                })
            
            conn.commit()
            return render_template('crm.html', companies=companies)
            
    except Exception as e:
        print(f"Error fetching companies: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to fetch companies", 
            "details": str(e)
        }), 500

@app.route('/api/companies', methods=['GET'])
def get_companies():
    """
    Fetches CRM dataset for API consumption.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT company_id, industry, company_size, annual_revenue, contract_status, region 
                FROM companies 
                LIMIT 50
            """)
            result = conn.execute(query)
            
            companies =[{"company_id": str(r[0]), "industry": r[1], "company_size": r[2], 
                          "annual_revenue": r[3], "contract_status": r[4], "region": r[5]} for r in result]
            
            conn.commit()
            return jsonify(companies)
            
    except Exception as e:
        return jsonify({"error": "Failed to fetch companies", "details": str(e)}), 500

@app.route('/api/add-company', methods=['POST'])
def add_company():
    """
    Handles adding a new B2B company.
    Takes unstructured 'sales_notes' from an employee and uses Gemini to map it to the CRM schema.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    sales_notes = request.form.get('sales_notes')
    if not sales_notes:
        return jsonify({"error": "No sales notes provided"}), 400

    prompt = """
    You are an intelligent CRM assistant. Analyze the sales rep's notes and extract B2B company details. 
    Return JSON strictly matching these keys (use "Unknown" if missing):
    {
        "industry": "string", "company_size": "string", "annual_revenue": "string",
        "region": "string", "contract_status": "string (e.g., Lead, Active, Churned)"
    }
    """
    
    try:
        # Use Gemini to parse unstructured notes into structured CRM data
        response = genai_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[sales_notes, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        profile = json.loads(response.text)
        
        # Generate a unique company ID
        company_id = "COMP-" + str(uuid.uuid4())[:8].upper()
        
        with engine.connect() as conn:
            # We construct a text block to embed based on key company attributes
            query = text("""
                INSERT INTO companies (
                    company_id, industry, company_size, annual_revenue, region, contract_status, company_vector
                )
                VALUES (
                    :cid, :ind, :size, :rev, :reg, :status, 
                    embedding('text-embedding-005', :ind || ' company in ' || :reg || ' with revenue ' || :rev)::vector
                ) 
                RETURNING company_id
            """)
            result = conn.execute(query, {
                "cid": company_id, 
                "ind": profile.get('industry', 'Unknown'),
                "size": profile.get('company_size', 'Unknown'),
                "rev": profile.get('annual_revenue', 'Unknown'),
                "reg": profile.get('region', 'Unknown'),
                "status": profile.get('contract_status', 'Lead')
            })
            new_cid = result.fetchone()[0]
            conn.commit()

        return jsonify({
            "status": "success",
            "company_id": str(new_cid),
            "parsed_profile": profile
        })

    except Exception as e:
        print(f"Error adding company: {traceback.format_exc()}")
        return jsonify({"error": "Operation Failed", "details": str(e)}), 500

@app.route('/api/update-company', methods=['POST'])
def update_company():
    """
    Allows employees to update specific B2B CRM fields (e.g., changing contract status, logging recent purchases).
    Replaces the old interaction / swipe mechanism.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    data = request.json
    company_id = data.get('company_id')
    
    # Fields that employees frequently update
    contract_status = data.get('contract_status')
    payment_behavior = data.get('payment_behavior')
    last_product_1 = data.get('last_product_1')

    if not company_id:
        return jsonify({"error": "company_id is required"}), 400

    try:
        with engine.connect() as conn:
            # Dynamically update provided fields (for brevity, explicitly updating standard ones)
            update_query = text("""
                UPDATE companies 
                SET contract_status = COALESCE(:status, contract_status),
                    payment_behavior = COALESCE(:payment, payment_behavior),
                    last_product_1 = COALESCE(:product, last_product_1)
                WHERE company_id = :cid
                RETURNING company_id, contract_status
            """)
            res = conn.execute(update_query, {
                "cid": company_id,
                "status": contract_status,
                "payment": payment_behavior,
                "product": last_product_1
            }).fetchone()
            
            conn.commit()
            
            if res:
                return jsonify({
                    "status": "success",
                    "company_id": res[0],
                    "new_contract_status": res[1]
                })
            else:
                return jsonify({"error": "Company not found"}), 404

    except Exception as e:
        print(f"Update error: {traceback.format_exc()}")
        return jsonify({"error": "Database error during update", "details": str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_companies():
    """
    Performs semantic vector search on the B2B CRM dataset using pgvector.
    Allows reps to search: "large tech companies in EMEA with high revenue"
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    query_text = request.args.get('query')
    if not query_text:
        return jsonify([])

    try:
        with engine.connect() as conn:
            print(f"Searching companies for: {query_text}")

            # Using Cosine Distance (<=>) for similarity
            # Combining pgvector distance with pg_ai natural language filtering
            search_sql = text("""
                SELECT company_id, industry, company_size, annual_revenue, region,
                       1 - (company_vector <=> embedding('text-embedding-005', :query)::vector) as score
                FROM companies 
                WHERE company_vector IS NOT NULL 
                AND ai.if(
                    prompt => 'Does this company profile: "Industry: ' || industry || ', Region: ' || region || '" logically fit the user search: "' ||  :query || '", at least 60%?',
                    model_id => 'gemini-3-flash-preview')  
                ORDER BY company_vector <=> embedding('text-embedding-005', :query)::vector
                LIMIT 10
            """)
            result = conn.execute(search_sql, {"query": query_text})
            
            hits =[]
            for row in result:
                hits.append({
                    "company_id": str(row[0]),
                    "industry": row[1],
                    "company_size": row[2],
                    "annual_revenue": row[3],
                    "region": row[4],
                    "score": round(float(row[5]), 3)
                })
            return jsonify(hits)
    except Exception as e:
        print(f"Error during search: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    # Using threaded=True to handle multiple concurrent requests better
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), threaded=True)
