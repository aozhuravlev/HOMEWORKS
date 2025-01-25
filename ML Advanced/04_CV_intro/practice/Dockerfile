FROM python:3.10

COPY requirements.txt /workdir/
COPY myapp/ /workdir/myapp/
COPY static/ /workdir/static/
COPY data/emnist-balanced-mapping.txt /workdir/

WORKDIR /workdir

RUN pip install -r requirements.txt

# Run the application
CMD ["uvicorn", "myapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
