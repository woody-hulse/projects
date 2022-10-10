################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/connect\ four.cpp 

OBJS += \
./src/connect\ four.o 

CPP_DEPS += \
./src/connect\ four.d 


# Each subdirectory must supply rules for building sources it contributes
src/connect\ four.o: ../src/connect\ four.cpp src/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/connect four.d" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


