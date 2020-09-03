## Jupyter Notebook over SSH
On remote computer:
```
jupyter notebook --no-browser --port=8887
```
On local computer:
```
ssh -N -L localhost:8888:localhost:8887 jonassl@owl25.uio.no
```
Open in browser:
```
http://localhost:8888
```