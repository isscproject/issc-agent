FROM python:3.6

# On Server
# ENV http_proxy http://10.61.3.150:8088
# ENV https_proxy http://10.61.3.150:8088
# ENV no_proxy 127.0.0.1,/var/run/docker.sock

ADD ./issc_agent /agent
ADD ./playground /playground

# @TODO to be replaced with `pip install pommerman`
#ADD . /pommerman
RUN cd /playground && pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
RUN cd /agent && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
#RUN cd /pommerman && python setup.py install
# end @TODO

EXPOSE 10080

ENV NAME Agent
# sudo docker build -t openrl/issc_agent:v0.1.0 .
# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]