COMPILER = gcc
CFLAGS   = -O2 -MMD -MP -Wall -Wextra
LDFLAGS  = -lm
INCLUDE  = -I./include
TARGET   = householder-qr
SRCDIR   = ./src
OBJDIR   = ./obj
SOURCES  = $(wildcard $(SRCDIR)/*.c)
OBJECTS  = $(addprefix $(OBJDIR)/, $(notdir $(SOURCES:.c=.o)))
DEPENDS  = $(OBJECTS:.o=.d)

.PHONY: all clean

all: $(TARGET)

-include $(DEPENDS)

$(TARGET): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@if [ ! -e $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(COMPILER) -o $@ $(CFLAGS) $(INCLUDE) -c $<

clean:
	rm -rf $(OBJECTS) $(DEPENDS) $(TARGET) $(OBJDIR)
	rm -f *~

