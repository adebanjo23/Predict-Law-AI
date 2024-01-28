from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langdetect import detect, LangDetectException
from googletrans import Translator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

app = Flask(__name__)
load_dotenv()
UPLOAD_FOLDER = 'pdfs'  # Folder to store uploaded PDFs
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS or filename.lower().endswith('.json'))

def error_response(message, status_code):
    return jsonify({'error': message}), status_code

@app.route('/upload', methods=['POST'])
def upload_pdf():
    username = request.form.get('username')
    pdf = request.files.get('pdf')

    # Check if username and pdf are provided
    if not (username and pdf):
        return error_response('Username and PDF file are required', 400)

    # Check if the file type is allowed
    if not allowed_file(pdf.filename):
        return error_response('Invalid file type. Only PDF files are allowed', 400)

    # Check if a file with the same username already exists
    existing_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{username}.pdf")
    if os.path.exists(existing_file_path):
        return error_response(f'A PDF file with the username {username} already exists. Delete it to upload a new one.', 400)

    # Save the PDF file
    filename = f"{username}.pdf"
    pdf.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'message': 'PDF uploaded successfully'}), 200

@app.route('/query', methods=['POST'])
def query_pdf():
    username = request.json.get('username')
    query = request.json.get('question')

    if not (username and query):
        return error_response('Username and question are required', 400)

    filename = f"{username}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(pdf_path):
        try:
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Detect language
            try:
                detected_language = detect(text)
            except LangDetectException as e:
                return error_response(f'Language detection error: {str(e)}', 500)

            if detected_language != 'en':
                try:
                    translator = Translator()
                    translated_text = translator.translate(text, src=detected_language, dest='en').text
                    text = translated_text
                except Exception as e:
                    print(f"Translation error: {str(e)}")

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Load or create embeddings
            store_name = username
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            # Query the model
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.8)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            return jsonify({'response': response}), 200
        except Exception as e:
            return error_response(str(e), 500)
    else:
        return error_response('PDF not found for the given username', 404)

@app.route('/delete', methods=['POST'])
def delete_pdf():
    username = request.json.get('username')

    if not username:
        return error_response('Username is required for deletion', 400)

    filename = f"{username}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
            return jsonify({'message': f'PDF file for {username} deleted successfully'}), 200
        except Exception as e:
            return error_response(str(e), 500)
    else:
        return error_response('PDF not found for the given username', 404)


if __name__ == '__main__':
    app.run(debug=True)
