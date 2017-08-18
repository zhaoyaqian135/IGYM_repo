# Simple Makefile
CXX = g++
PROJ = multiplayer
PROJ1 = videocapture
PROJ2 = simulation
PROJ3 = oneplayer
EDITOR = gvim

all: $(PROJ) $(PROJ1) $(PROJ2)
$(PROJ): $(PROJ).cpp
	$(CXX) -g `pkg-config --cflags opencv` -std=c++11 -o $(PROJ) $(PROJ).cpp  `pkg-config --libs opencv` -lpthread

$(PROJ1): $(PROJ1).cpp
	$(CXX) -g `pkg-config --cflags opencv` -std=c++11 -o $(PROJ1) $(PROJ1).cpp  `pkg-config --libs opencv` -lpthread

$(PROJ2): $(PROJ2).cpp
	$(CXX) -g `pkg-config --cflags opencv` -std=c++11 -o $(PROJ2) $(PROJ2).cpp  `pkg-config --libs opencv` -lpthread

$(PROJ3): $(PROJ3).cpp
	$(CXX) -g `pkg-config --cflags opencv` -std=c++11 -o $(PROJ3) $(PROJ3).cpp  `pkg-config --libs opencv` -lpthread

clean:
	-rm -f *.o core $(PROJ)
	-rm -f *.o core $(PROJ1)
	-rm -f *.o core $(PROJ2)
	-rm -f *.o core $(PROJ3)

#-framework GLUT -framework OpenGL


