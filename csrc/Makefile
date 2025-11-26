CC = gcc
CFLAGS = -O3 -fPIC -march=native
LDFLAGS = -shared

all: libfast_update.so

libfast_update.so: fast_update.o
    $(CC) $(LDFLAGS) -o libfast_update.so fast_update.o

fast_update.o: fast_update.c fast_update.h
    $(CC) $(CFLAGS) -c fast_update.c

clean:
    rm -f *.o *.so
