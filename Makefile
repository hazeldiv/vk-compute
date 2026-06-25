CC           := gcc
CFLAGS	     := -O2 -Wall -Wextra -Iinclude -I"$(VULKAN_SDK)\Include"
LDFLAGS      := -L"$(VULKAN_SDK)\Lib" -lvulkan-1 -luser32 -lgdi32
TARGET       := main
SRC_DIR      := src
BUILD_DIR    := build
BIN_DIR      := bin
SHADER_DIR   := shader

SRCS         := $(wildcard $(SRC_DIR)/*.c)
OBJS         := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS         := $(OBJS:.o=.d)
SHADERS      := $(wildcard $(SHADER_DIR)/*.comp)
SHADERS_OBJS := $(SHADERS:$(SHADER_DIR)/%.comp=$(BIN_DIR)/%.spv)

all: ${SHADERS_OBJS} $(BIN_DIR)/$(TARGET)

$(BIN_DIR)/$(TARGET): $(OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built successfully: $@.exe"
	
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR)/%.spv: $(SHADER_DIR)/%.comp
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	"$(VULKAN_SDK)\Bin\glslangValidator" -V --target-env vulkan1.1 $< -o $@

-include $(DEPS)

run: all
	@cd bin && main.exe

clean:
	@if exist bin del /Q /F bin\*
	@if exist bin rmdir bin
	@if exist build del /Q /F build\*
	@if exist build rmdir build
