#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// I2C LCD nesnesi oluştur (adres genellikle 0x27 veya 0x3F)
LiquidCrystal_I2C lcd(0x27, 16, 2);  // Adres, sütun, satır

String receivedString;
float receivedFloat;
bool newData = false;

void setup() {
  // LCD'yi başlat
  lcd.init();
  lcd.backlight();
  lcd.print("Float Receiver");
  
  // Seri iletişimi başlat
  Serial.begin(9600);
}

void loop() {
  receiveFloat();
  
  if (newData) {
    lcd.clear();
    if (receivedFloat == 0.0) {
      lcd.setCursor(0, 0);
      lcd.print("Stay");
    }
    else if (receivedFloat < 0.0) {
      lcd.setCursor(0, 0);
      lcd.print("Go left");
      lcd.setCursor(0, 1);
      lcd.print(abs(receivedFloat), 4);
      lcd.print("cm");
    }
    else {
      lcd.setCursor(0, 0);
      lcd.print("Go right");
      lcd.setCursor(0, 1);
      lcd.print(receivedFloat, 4);
      lcd.print("cm");
    }  // 4 ondalık basamak
    
    newData = false;
  }
}

void receiveFloat() {
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;
  
  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();

    if (rc != endMarker) {
      receivedString += rc;
    }
    else {
      receivedFloat = receivedString.toFloat();
      receivedString = "";
      newData = true;
    }
  }
}