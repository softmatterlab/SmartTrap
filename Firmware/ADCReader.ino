/*
Arduino sketch for testing and reading the AD7616 using an arduino portenta.
*/
#define SEQEN       PI_10  // UART 0 RTS
#define RESET       PI_13  // UART 0 CTS
//#define REFSEL      PI_15
#define WR_BURST    PH_10
#define SCLK_RD     PH_12
#define CS          PI_4


#define CHSEL0      PI_7
#define CHSEL1      PI_5
#define CHSEL2      PC_13

#define HW_RNGSEL0  PA_4

 // CHanged here to CAN1_RX from PA_4 and previously PC_13. Should be PB_8 on breakout!
#define AD_BUSY     PI_6
#define CONVST      PH_6 // PA_6 before(on breakout?)

#define DB0         PB_15
#define DB1         PH_9
#define DB2         PB_3
#define DB3         PH_11
#define DB4         PB_4
#define DB5         PH_14
#define DB6         PB_14
#define DB7         PJ_8
#define DB8         PD_7
#define DB9         PA_11 // USB D- on breakout
#define DB10        PD_6
#define DB11        PG_14 // UART2_TX
#define DB12        PD_3
#define DB13        PG_9  //UART2_RX
#define DB14        PB_9
#define DB15        PI_2


uint8_t busy_counter = 0;

// Ser/Par (pin 40 of AD7616) connected to logic low,
/*
F_A_A referce to the OP-amp connected to force detector A and the op-amp of the 4131 circuit.

Source   Channel no   ADC_results index
PD_A     0A                   // Photodiode
F_A_SUM  1A                 
F_A_Y    2A                 
F_A_X    3A                 
P_A_SUM  4A                   // Position A sum
P_A_Y    5A                 
P_A_X    6A                 
TEMP_A   7A                   // Temperature

PD_A     0B                 
F_B_SUM  1B                 
F_B_Y    2B                 
F_B_X    3B                 
P_B_SUM  4B                 
P_B_Y    5B                 
P_B_X    6B                 
TEMP_B   7B                 
*/

struct ADCResult {
  uint16_t first;
  uint16_t second;
};

void setup_ADC() {
  pinMode(LED_BUILTIN, OUTPUT);

  // Initalize the pins to output/input
  pinMode(SEQEN, OUTPUT);
  pinMode(RESET, OUTPUT);
  pinMode(WR_BURST, OUTPUT);
  pinMode(SCLK_RD, OUTPUT);
  pinMode(CS, OUTPUT);

  pinMode(HW_RNGSEL0, OUTPUT);

  pinMode(CHSEL0, OUTPUT);
  pinMode(CHSEL1, OUTPUT);
  pinMode(CHSEL2, OUTPUT);
  pinMode(AD_BUSY, INPUT);
  pinMode(CONVST, OUTPUT);

  // Set the paralell input pins
  pinMode(DB0,INPUT_PULLUP);  // Used to have just input on all these
  pinMode(DB1,INPUT_PULLUP);
  pinMode(DB2,INPUT_PULLUP);
  pinMode(DB3,INPUT_PULLUP);
  pinMode(DB4,INPUT_PULLUP);
  pinMode(DB5,INPUT_PULLUP);
  pinMode(DB6,INPUT_PULLUP);
  pinMode(DB7,INPUT_PULLUP);
  pinMode(DB8,INPUT_PULLUP);
  pinMode(DB9,INPUT_PULLUP);
  pinMode(DB10,INPUT_PULLUP);
  pinMode(DB11,INPUT_PULLUP);
  pinMode(DB12,INPUT_PULLUP);
  pinMode(DB13,INPUT_PULLUP);
  pinMode(DB14,INPUT_PULLUP);
  pinMode(DB15,INPUT_PULLUP);

  // Select channel 0
  digitalWrite(CHSEL0,LOW);
  digitalWrite(CHSEL1,LOW);
  digitalWrite(CHSEL2,LOW);

  // Set read range of the ADC to +- 10V, (HIGH), for +- 5V Set low.
  digitalWrite(HW_RNGSEL0, HIGH);

  //Set RD and CS high(default mode)
  digitalWrite(SCLK_RD,HIGH);
  digitalWrite(CS,HIGH);

  // digitalWrite(REFSEL, HIGH);  // Internal reference, Tied high on board
  digitalWrite(WR_BURST, LOW); // ?
  digitalWrite(SEQEN, LOW); // Enable sequencer, hardware mode only

  // Activate the ADC pin by setting the reset pin high after having had it low
  digitalWrite(RESET, LOW);
  delay(100);
  digitalWrite(RESET, HIGH);
}
void sampleDataUltra(int* data_array, uint8_t index){
  //int value = 0;
  data_array[index] = 0;
  data_array[index] |= ((GPIOB->IDR & (1 << 15)) >> 15) << 0;
  data_array[index] |= ((GPIOH->IDR & (1 << 9)) >> 9) << 1;
  data_array[index] |= ((GPIOB->IDR & (1 << 3)) >> 3) << 2;
  data_array[index] |= ((GPIOH->IDR & (1 << 11)) >> 11) << 3;
  data_array[index] |= ((GPIOB->IDR & (1 << 4)) >> 4) << 4;
  data_array[index] |= ((GPIOH->IDR & (1 << 14)) >> 14) << 5;
  data_array[index] |= ((GPIOB->IDR & (1 << 14)) >> 14) << 6;
  data_array[index] |= ((GPIOJ->IDR & (1 << 8)) >> 8) << 7;
  data_array[index] |= ((GPIOD->IDR & (1 << 7)) >> 7) << 8;
  data_array[index] |= ((GPIOA->IDR & (1 << 11)) >> 11) << 9;
  data_array[index] |= ((GPIOD->IDR & (1 << 6)) >> 6) << 10;
  data_array[index] |= ((GPIOG->IDR & (1 << 14)) >> 14) << 11;
  data_array[index] |= ((GPIOD->IDR & (1 << 3)) >> 3) << 12;
  data_array[index] |= ((GPIOG->IDR & (1 << 9)) >> 9) << 13;
  data_array[index] |= ((GPIOB->IDR & (1 << 9)) >> 9) << 14;

  if (((GPIOI->IDR & (1 << 2)) >> 2)){
    data_array[index] -= 32768;
  }
}


void sampleDataUint8(uint8_t* data_array, uint16_t msb_idx, uint16_t lsb_idx){
  //Set values to 0
  data_array[lsb_idx] = 0; // TODO check if this is necessary
  data_array[msb_idx] = 0;

  // Set the LSB
  data_array[lsb_idx] |= ((GPIOB->IDR & (1 << 15)) >> 15) << 0;
  data_array[lsb_idx] |= ((GPIOH->IDR & (1 << 9)) >> 9) << 1;
  data_array[lsb_idx] |= ((GPIOB->IDR & (1 << 3)) >> 3) << 2;
  data_array[lsb_idx] |= ((GPIOH->IDR & (1 << 11)) >> 11) << 3;
  data_array[lsb_idx] |= ((GPIOB->IDR & (1 << 4)) >> 4) << 4;
  data_array[lsb_idx] |= ((GPIOH->IDR & (1 << 14)) >> 14) << 5;
  data_array[lsb_idx] |= ((GPIOB->IDR & (1 << 14)) >> 14) << 6;
  data_array[lsb_idx] |= ((GPIOJ->IDR & (1 << 8)) >> 8) << 7;

  // Set the MSB
  data_array[msb_idx] |= ((GPIOD->IDR & (1 << 7)) >> 7) << 0;
  data_array[msb_idx] |= ((GPIOA->IDR & (1 << 11)) >> 11) << 1;
  data_array[msb_idx] |= ((GPIOD->IDR & (1 << 6)) >> 6) << 2;
  data_array[msb_idx] |= ((GPIOG->IDR & (1 << 14)) >> 14) << 3;
  data_array[msb_idx] |= ((GPIOD->IDR & (1 << 3)) >> 3) << 4;
  data_array[msb_idx] |= ((GPIOG->IDR & (1 << 9)) >> 9) << 5;
  data_array[msb_idx] |= ((GPIOB->IDR & (1 << 9)) >> 9) << 6;
  if (((GPIOI->IDR & (1 << 2)) >> 2) == 0) {
        data_array[msb_idx] |= 1 << 7; // Set the flag/sign bit
    }
  
}

void select_channel_fast(uint8_t channel_pair) {
  switch (channel_pair) {
    case 0:
      GPIOI->BSRR = (1 << (7 + 16));
      GPIOI->BSRR = (1 << (5 + 16));
      GPIOC->BSRR = (1 << (13 + 16));
      break;

    case 1:
      GPIOI->BSRR = (1 << 7);
      GPIOI->BSRR = (1 << (5 + 16));
      GPIOC->BSRR = (1 << (13 + 16));
      break;

    case 2:
      GPIOI->BSRR = (1 << (7 + 16));
      GPIOI->BSRR = (1 << 5);
      GPIOC->BSRR = (1 << (13 + 16));
      break;

    case 3:
      GPIOI->BSRR = (1 << 7);
      GPIOI->BSRR = (1 << 5);
      GPIOC->BSRR = (1 << (13 + 16));
      break;

    case 4:
      GPIOI->BSRR = (1 << (7 + 16));
      GPIOI->BSRR = (1 << (5 + 16));
      GPIOC->BSRR = (1 << 13);
      break;

    case 5:
      GPIOI->BSRR = (1 << 7);
      GPIOI->BSRR = (1 << (5 + 16));
      GPIOC->BSRR = (1 << 13);
      break;

    case 6:
      GPIOI->BSRR = (1 << (7 + 16));
      GPIOI->BSRR = (1 << 5);
      GPIOC->BSRR = (1 << 13);
      break;

    case 7:
      GPIOI->BSRR = (1 << 7);
      GPIOI->BSRR = (1 << 5);
      GPIOC->BSRR = (1 << 13);
      break;

    default:
      return;
  }
}


void read_adc_fast_return(uint8_t channel_pair, int* data_array, uint8_t index){
  if (channel_pair>7){
    return;
  }
  // Select the channel to sample during the next cycle(not this one)
  select_channel_fast((channel_pair+1)%8);
  busy_counter = 0;
  
  GPIOH->BSRR = (1 << 6);
  GPIOH->BSRR = (1 << (6 + 16));
  uint8_t lim = 0;
  while (((GPIOI->IDR & (1 << 6)) >> 6) && lim<200){
    lim+=1;
  }
  
  GPIOI->BSRR = (1 << (4 + 16));
  GPIOH->BSRR = (1 << (12 + 16));

  // TODO put the data directly in the actual return array later(as 8bit numbers).
  sampleDataUltra(data_array, index);

  GPIOH->BSRR = (1 << 12);
  GPIOI->BSRR = (1 << 4);

  GPIOI->BSRR = (1 << (4 + 16));
  GPIOH->BSRR = (1 << (12 + 16));

  sampleDataUltra(data_array, index+1);
  GPIOI->BSRR = (1 << 4);
  GPIOH->BSRR = (1 << 12);
}

void read_adc_2_uint8_t(uint8_t channel_pair, uint8_t* data_array, uint16_t index1, uint16_t index2){
  if (channel_pair>7){
    return;
  }
  // Select the channel to sample during the next cycle(not this one)
  select_channel_fast((channel_pair+1)%8);
  busy_counter = 0;
  
  GPIOH->BSRR = (1 << 6);
  GPIOH->BSRR = (1 << (6 + 16));
  uint8_t lim = 0;
  while (((GPIOI->IDR & (1 << 6)) >> 6) && lim<200){
    lim+=1;
  }
  
  GPIOI->BSRR = (1 << (4 + 16));
  GPIOH->BSRR = (1 << (12 + 16));

  // TODO put the data directly in the actual return array later(as 8bit numbers).
  sampleDataUint8(data_array, index1+1, index1);

  GPIOH->BSRR = (1 << 12);
  GPIOI->BSRR = (1 << 4);

  GPIOI->BSRR = (1 << (4 + 16));
  GPIOH->BSRR = (1 << (12 + 16));

  sampleDataUint8(data_array, index2+1, index2);
  GPIOI->BSRR = (1 << 4);
  GPIOH->BSRR = (1 << 12);
}
