from io import BytesIO
from docker import APIClient

dockerfile = '''
FROM python:3.9

WORKDIR /home

ADD C:/Users/Quint/git/codegenie/dockerfiles/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY C:/Users/Quint/git/codegenie/dockerfiles/ /home

CMD [ "python", "/home/test.py" ]
'''

f = BytesIO(dockerfile.encode('utf-8'))
cli = APIClient(base_url='npipe:////./pipe/docker_engine')
[print(line) for line in cli.build(fileobj=f, rm=True, tag='mypython/test')]

# client = docker.DockerClient(base_url='npipe:////./pipe/docker_engine')

# path="./dockerfiles"

# client.images.build(path=path, tag='mypython', rm=True)

# client.containers.run('mypython', name='mycontainer', ports={'5000/tcp': 5000})