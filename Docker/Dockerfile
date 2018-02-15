FROM ubuntu:16.04

RUN apt-get update && apt-get install -y sudo apt-utils && rm -rf /var/lib/apt/lists/*

# Set the working directory 
WORKDIR /app/

RUN apt-get update
RUN apt-get install -y git dialog 

# torch
RUN git clone https://github.com/torch/distro.git ~/torch --recursive && \
    cd ~/torch && \
   	bash install-deps && \
    bash install.sh -b

ENV PATH="/root/torch/install/bin/:${PATH}"

RUN luarocks install lzmq
RUN luarocks install "https://raw.github.com/Neopallium/lua-pb/master/lua-pb-scm-0.rockspec"

#synching data
RUN apt-get install -y python-pip
RUN pip install awscli


RUN git clone https://github.com/tddebruin/dlforce-memexp-torch.git /app

WORKDIR /app/

# Run torch experiments when the container launches
CMD ["sh", "start.sh"]