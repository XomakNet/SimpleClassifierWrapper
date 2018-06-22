FROM tiangolo/uwsgi-nginx-flask:python3.6
COPY ./ /app/
COPY docker/uwsgi.ini /app/
RUN apt-get update && apt-get install -y unixodbc-dev
RUN pip install -r requirements.txt