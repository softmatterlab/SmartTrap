/*
Note:
This script has been tested and works as is for the arduino portenta and the dac8554.

*/

//#define SPI_SYNC PJ_9 //UART3 RX PIN, sync pin
#define LDAC	   PE_2 // Needs to be put low, used if all outputs are to be changed at the same time
#define IOVDD_DAC PI_3 // external power 
#define SPI_SYNC PJ_9 //UART3 RX PIN
#define AUTO_ALIGN_LIM 0  // 40 is standard, seem to work fine. Used 20 before

#define AX  0.0501 // TODO update these
#define AY  0.0513
#define BX  0.0551
#define BY  0.0574

#define ABX = 0.91
#define ABY = 0.89
#define BAX = 1.10
#define BAY = 1.12

//float dist_x_f = 0;
//float dist_y_f = 0;
//float autoalign_fac_float = 500; // Estimated a decent value for this.

//uint8_t align_counter = 0;
//uint8_t align_limit = 5;  // 5 was a good value before, maybe too low now that the code is faster
float autoalign_fac = 0.0002;//5/60000;//0.3 before; // Taking into account that we do an averaging now. // removed factor 40 below and decreased from 0.2 to (which would have yielded 8) to 5
float local_align_fac = 0;
uint16_t trap_val = 32768;
uint8_t position_move_dir = 0;
long move_counter = 0;  // X, Y, SUM
//float current_force = 0;


SPI spi(PC_3, PC_2, PI_1);

void setup_DAC() {
  pinMode(SPI_SYNC, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  GPIOJ->MODER |= (1 << (9 * 2)); // set sync as output pin

  pinMode(IOVDD_DAC,OUTPUT);
  digitalWrite(IOVDD_DAC, HIGH);
  pinMode(LDAC, OUTPUT);
  digitalWrite(LDAC, LOW);
  spi.frequency(25000000);           // Set up your frequency. Max is 25 MHz according to datasheet when operating at a logic voltage level of 3.4V as the portenta does.
  spi.format(8, 3);                  // Messege length (bits), SPI_MODE - check these in your SPI decice's data sheet.
}

void writeNumberFast(uint8_t DACData[]){
  // Writes a number to the DAC
  // Rather fast, at 25 MHz we can write all the 4 channels in ca 10 microseconds.
  GPIOJ->ODR &= ~(1 << 9); // set sync low
  spi.write((const char*)DACData, 3, NULL, 0);
  GPIOJ->ODR |= (1 << 9); // set sync high
}

void autoalign(uint8_t trap){
  // Trap = 0 => move trap A to match trap B
  // TODO make this work with the 8bit arrays as well.
  local_align_fac = autoalign_fac*((PSD_Force_A[2]-32768) + (PSD_Force_B[2]-32768));
  // TODO maybe take intensity of signal into account by incorporating the sum reading as well.
  dist_x = Calibration_Factors[0]*float(PSD_Force_A[0]-32768) + Calibration_Factors[1] * float(PSD_Force_B[0]-32768); // + instead of -?
  dist_y = Calibration_Factors[2]*float(PSD_Force_A[1]-32768) - Calibration_Factors[3] * float(PSD_Force_B[1]-32768);
  if (trap==0){

    // Move A to reduce the distance between A and B
    // Check if we can get an overflow on this, seems like it may be possible.

    //----------------------AX------------------------
    trap_val = (DAC_AX[1] << 8) | DAC_AX[2];
    // Fixing risk of overflow.
    int movement = int(local_align_fac*float(dist_x));
    if (long(trap_val + movement) > 65535){
        trap_val = 65535;
    }
    else if (long(trap_val + movement)<0){
      trap_val = 0;
    }
    else{
      // if (movement*movement<1600){// error here since we wont move for very large movements
      trap_val += movement;
    }
    DAC_AX[1] = (trap_val >> 8) & 0xFF;
    DAC_AX[2] =  trap_val & 0xFF;

    //-----------------------AY------------------------    
    trap_val = (DAC_AY[1] << 8) | DAC_AY[2];

    // Fixing risk of overflow.
    movement = int(local_align_fac*float(dist_y));
    if (long(trap_val - movement) > 65535){
        trap_val = 65535;
    }
    else if (long(trap_val - movement) < 0){
      trap_val = 0;
    }
    else{
      trap_val -= movement;
    }

    DAC_AY[1] = (trap_val >> 8) & 0xFF;
    DAC_AY[2] =  trap_val & 0xFF;
  
  }
  else{
    //------------------BX-----------------------------
    // TODO autoalign B does not work, perhaps due to error in signs.
    
    trap_val = (DAC_BX[1] << 8) | DAC_BX[2];
    // Fixing risk of overflow.
    int movement = int(local_align_fac*float(dist_x));
    // Changed sign here back to + for x, check if it works now
    if (long(trap_val + movement) > 65535){
        trap_val = 65535;
    }
    else if (long(trap_val + movement)<0){
      trap_val = 0;
    }
    else{
      trap_val += movement;
    }
    DAC_BX[1] = (trap_val >> 8) & 0xFF;
    DAC_BX[2] =  trap_val & 0xFF;

    //------------------BY-----------------------------
    trap_val = (DAC_BY[1] << 8) | DAC_BY[2];
    // Changed sign here to +
    // Fixing risk of overflow.
    movement = int(local_align_fac*float(dist_y));

    if (long(trap_val + movement) > 65535){
        trap_val = 65535;
    }
    else if (long(trap_val + movement) < 0){
      trap_val = 0;
    }
    else{
      trap_val += movement;
    }
    DAC_BY[1] = (trap_val >> 8) & 0xFF;
    DAC_BY[2] =  trap_val & 0xFF;

  }
}

void constant_speed_move(uint8_t DAC_DATA[], uint16_t stepsize, uint16_t limits[]){
  /*
  Moves the trap in DAC_data between upper limits[0] and lower limits[1] on the movement range
  in steps of stepsize. 
  */
  move_counter += 1;
  if(move_counter & 15){
    return ;
  }
  uint16_t DAC_pos = ((DAC_DATA[1] << 8) | DAC_DATA[2]);

  if (position_move_dir == 0) {
      // Move forward
      DAC_pos += stepsize;
      if (DAC_pos >= limits[0]) {
        // Removed the limit check to ensure that we can start from any position
        //DAC_pos = limits[0];
        position_move_dir = 1;
      }
  }
  else {
      DAC_pos -= stepsize;
      if (DAC_pos <= limits[1]) { // The second condition checks for underflow.
        //DAC_pos = limits[1];
        position_move_dir = 0;
      }
  }

  DAC_DATA[1] = (DAC_pos >> 8) & 0xFF;
  DAC_DATA[2] = DAC_pos & 0xFF;

}


void constant_speed_move_force_lims_protocol(uint8_t DAC_DATA[], uint16_t stepsize, uint16_t force_reading_limits[], uint16_t force){
  // Moves back and forth between force limits. 
  move_counter += 1;
  if(move_counter & 15){
    return ;
  }
  uint16_t DAC_pos = ((DAC_DATA[1] << 8) | DAC_DATA[2]);

  if (position_move_dir == 0) {
      // Move forward
      DAC_pos += stepsize;

      if (DAC_pos >= limits[0]) {
        position_move_dir = 1;
      }
  }
  else {
      DAC_pos -= stepsize;
      if (DAC_pos <= limits[1]) { // The second condition checks for underflow.
        //DAC_pos = limits[1];
        position_move_dir = 0;
      }
  }

  DAC_DATA[1] = (DAC_pos >> 8) & 0xFF;
  DAC_DATA[2] = DAC_pos & 0xFF;

}

void constant_speed_move_force_lims(uint8_t DAC_DATA[], uint16_t stepsize, uint16_t force_reading_limits[], uint8_t axis){
  // TODO NOT FINISHED YET!
  // Currently moves in one direction until a force threshold is exceeded or a position limit is reached, position limit is fixed.
  move_counter += 1;
  if(move_counter & 15){
    return ;
  }
  uint16_t DAC_pos = ((DAC_DATA[1] << 8) | DAC_DATA[2]);

  if (axis == 0||axis==2) {
      // Move forward in x or y by increasing the value of the reading, 0 for x, 2 for y
      if (DAC_pos <= 62000 && !force_limit_reached) {
        DAC_pos += stepsize;    
        //force_limit_reached = false;
      }
      
  }
  else{
      // move backwards, 1 for x, 3 for y.
      if (DAC_pos >= 2000 && !force_limit_reached) {
        DAC_pos -= stepsize; 
      }      
  }

  DAC_DATA[1] = (DAC_pos >> 8) & 0xFF;
  DAC_DATA[2] = DAC_pos & 0xFF;
}


void constant_speed_move_force_lims_reverse(uint8_t DAC_DATA[], uint16_t stepsize, uint16_t force_reading_limits[], uint8_t axis){
  // TODO NOT FINISHED YET!
  // Currently moves in one direction until a force threshold is exceeded or a position limit is reached, position limit is fixed.
  move_counter += 1;
  if(move_counter & 15){
    return ;
  }
  uint16_t DAC_pos = ((DAC_DATA[1] << 8) | DAC_DATA[2]);

  if (axis == 0||axis==2) {
      // Move forward in x or y by increasing the value of the reading, 0 for x, 2 for y
      if (DAC_pos <= 62000 && !force_limit_reached) {
        DAC_pos += stepsize;    
        //force_limit_reached = false;
      }
      else if (DAC_pos >1000 && force_limit_reached) {
        DAC_pos -= force_reading_limits[1];    
      }
      
  }
  else{
      // move backwards, 1 for x, 3 for y.
      if (DAC_pos >= 2000 && !force_limit_reached) {
        DAC_pos -= stepsize; 
      }
      if (DAC_pos <= 62000 && force_limit_reached) {
        DAC_pos += force_reading_limits[1]; 
      }
  }

  DAC_DATA[1] = (DAC_pos >> 8) & 0xFF;
  DAC_DATA[2] = DAC_pos & 0xFF;
}

