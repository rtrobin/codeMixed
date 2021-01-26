# celery sample code

Run rabbitmq as broker

```
docker run -d --rm --name=rabbitmq -p 5672:5672 rabbitmq:alpine
```

Run redis as result backend

```
docker run -d --rm --name=redis -p 6379:6379 redis:alpine
```

Run celery worker

```
celery -A proj.tasks worker --loglevel=INFO
```

Run celery test script

```
python main.py
```