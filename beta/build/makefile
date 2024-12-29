CC := gcc
CFLAGS := -Wall -O3
INCLUDE := -I./../common/
LIBS := -lm

SRCS := $(wildcard ./../common/*.c)
SRCS += ./../src/main.c
OBJS = $(SRCS:.c=.o)

TARGET := exec

$(TARGET): $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $(OBJS) $(LIBS)
	rm -f $(OBJS)

%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@ $(LIBS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)