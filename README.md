# Setup

1. Create Virtual Environment
 - python -m venv venv
 - source venv/bin/activate

2. Install Dependencies
- pip install -r requirements.txt

3. Generate Fake Data
- python data_generator.py

4. Train the Model
- python train.py

5. Start the API
- uvicorn main:app --reload

6. Test the API
- [](http://127.0.0.1:8000/docs)