FROM jpfeil/hydra:0.3.1

MAINTAINER Jacob Pfeil, jpfeil@ucsc.edu

COPY typhon /opt/typhon
WORKDIR /data
ENTRYPOINT ["python", "/opt/typhon/run.py"]
CMD ["-h"]
