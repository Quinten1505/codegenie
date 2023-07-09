FROM python:3.9

WORKDIR /home

ADD ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ /home

CMD [ "python", "/home/test.py" ]