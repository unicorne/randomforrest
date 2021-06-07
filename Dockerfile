FROM python:3.9

WORKDIR /app

RUN apt update
RUN apt -y install python3-pip
RUN apt -y install screen


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED="true"

COPY . .

CMD ["python3", "-u", "./main.py" ]

