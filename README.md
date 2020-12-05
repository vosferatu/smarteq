# smarteq
**_Smart Equalizer_** implemented in _Python_ for Computer Based Sound Production Course at the Faculty of Computer and Information Science from the University of Ljubljana


## Topic

### Introduction 
Music comes in a wide range of **sounds** and **tastes**. We believe that an **automatic equalizer** that **determines the genre** and other details of a song using aspects like _frequency_ can improve the user experience when listening to music.

### Goals 
Our  goal  is  to  determine  the  genre  of  a  song  and  provide an equalizer that **enhances the sound experience for the user**.

### Technologies and Libraries
- **_Python_**
- _pyaudio_ for Audio
- _scipy_ for Math 
- _pyside_ for GUI

### Implementation
The  plan  is  to  divide  the  work  between  the  two  members of the group. One will explore the **genre determination** and the other the **equalizer**  itself  and  how  to  find  the  correct  settings  for  each song/sound. On a first phase we’ll build a simpler reduced version until we assure everything is working. On a second phase we’ll **widen the range of genres** and on a third phase we’ll try to make small improvements if we have time.

### Dataset

We chose the [GITZAN](http://opihi.cs.uvic.ca/sound/genres.tar.gz) dataset because it is relatively small (~1 _gigabyte_) compared to others like *FMA* (Free Music Archive), *RWC Music Database* or *MSD* (Million Song Dataset) which have hundreds of _gigabytes_. The dataset consists of *1000 audio tracks* each 30 seconds long. It has genres namely, blues, classical, country, disco, hiphop, jazz, reggae, rock, metal and pop with 100 songs per genre.



## Running

with python 3.7

cd smarteq/

source env/bin/activate

pip install -r requirements.txt

python genre.py aaaaa.wav   // python genre.py aaaaa.wav


x = genre
y = wav file
