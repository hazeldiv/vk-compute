CC           := gcc
CFLAGS	     := -O2 -Wall -Wextra -Iinclude -I"$(VULKAN_SDK)/Include"
LDFLAGS      := -L"$(VULKAN_SDK)/Lib" -lvulkan-1 -luser32 -lgdi32
TARGET       := main
SRC_DIR      := src
BUILD_DIR    := build
BIN_DIR      := bin
SHADER_DIR   := shader

SRCS         := $(wildcard $(SRC_DIR)/*.c)
OBJS         := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS         := $(OBJS:.o=.d)
rwildcard    = $(foreach d,$(wildcard $(1)*),$(call rwildcard,$(d)/,$(2)) $(filter $(subst *,%,$(2)),$(d)))
SHADERS      := $(call rwildcard,$(SHADER_DIR)/,*.comp)
SHADER_OUT   := $(BIN_DIR)/shader
SHADER_OUT_W := $(subst /,\,$(SHADER_OUT))
SHADERS_OBJS := $(addprefix $(SHADER_OUT)/,$(notdir $(SHADERS:.comp=.spv)))

all: ${SHADERS_OBJS} $(BIN_DIR)/$(TARGET)

$(BIN_DIR)/$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built successfully: $@.exe"
	
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

define COMPILE_SHADER
$(SHADER_OUT)/$(notdir $(basename $(1))).spv: $(1)
	@mkdir -p $(SHADER_OUT)
	"$(VULKAN_SDK)/Bin/glslangValidator" -V --target-env vulkan1.1 "$(1)" -o $$@
endef
$(foreach f,$(SHADERS),$(eval $(call COMPILE_SHADER,$(f))))

-include $(DEPS)

run: all
	@cd bin && main.exe

clean:
	@rm -rf $(BIN_DIR) $(BUILD_DIR)
