---
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: true
    author: Ceaser
---


ubuntu를 쓰면서 내가 필요하다고 느낀 부분들을 한꺼번에 정리하는 페이지이다. 
나는 지금 Ubuntu 20.03버전을 쓰고 있으며 Jammy버전을 쓰고 있다.  
그런데 아직까지 foscal 버전과 jammy 버전과의 차이는 무엇이 다른지는 잘 모르겠다.  
그리고 제일 필요한 wake on lan기능의 설정과 desktop remote control 기능의 설정이 필요하다.  
ssl을 설치하여 내가 자주 쓰는 Xshell과 통신하도록 한다.  

ubuntu를 설치하는 방법은 여기서 쓰지는 않겠다. Youtube나 아니면 다른 블로그에서도 찾아 볼 수 있기에 넘어간다.  

그 다음 부분이 내가 여기서 정리할 부분이다. 

```shell
sudo apt update
```
또는 
```shell
sudo apt-get update
```

내게는 첫번째로 Wake on lan기능이 필요하다. 어디서든 컴퓨터를 켜서 deep learning을 돌린 것을 확인하려 한다.  
그러려면 ubuntu에 패키지를 설치해 주어야 한다. 
```shell
sudo apt-get install net-tools wakeonlan ethtool
```
이를 설치해 주고는 ```ifconfig```으로 인터넷을 어떻게 연결하고 있는지 확인한다. 윈도우의 ipconfig와 비슷하지만 
f가 다르다는 것을 잊지 말자.   
실행하여 보면  
```shell
enp4s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 000.000.000  netmask 255.255.255.0  broadcast ---.---.--.---
        inet6 ----::----:----:----:----  prefixlen 64  scopeid 0x20<link>
        ether --:--:--:--:--:--  txqueuelen 1000  (Ethernet)
        RX packets 3398  bytes 3960078 (3.9 MB)
        RX errors 0  dropped 67  overruns 0  frame 0
        TX packets 1642  bytes 154058 (154.0 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device memory 0x82600000-826fffff
```
나의 메인보드의 경우에는 LAN선을 두개 꽃을 수 있는 곳이 있어 그런지 두개가 나온다. 그 중에서 **inet**과 **broadcast**가 
살아있는 것이 지금 사용하고 있는 lan card이다. (익명화를 위하여 필요한 정보는 가림.)  

ethtool으로 내가 쓰는 이 lan card가 어떤 셋팅을 하고 있는지 확인하여 보자.  
```shell
ethtool enp4s0
```
