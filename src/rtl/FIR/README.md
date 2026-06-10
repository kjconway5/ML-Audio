# Type I Symmetric FIR  Compensation Filter

Post PDM to PCM conversion using the CIC filter(an anti-aliasing and decimation filter). It is necessary to have a relatively flat pass-band gain and transition region when fed into a larger system. Due to the pass-band magnitude response of the CIC filter:

&emsp; &emsp; $\left| H_{CIC}(e^{j2 \pi f}) \right|=\left| \frac{sin(2\pi fD/2)}{sin(2\pi f/2)}\right|^M$

The CIC filter produces a pass-band droop as the frequency increases towards the transition band: 

**insert image**

To achieve a flatter pass-band gain and sharper transition region we can utilize a FIR filter with tap coefficients that contain the inverse magnitude response of the CIC filter. 

## Design Steps
1. Calculate the inverse magnitude response:
&emsp; &emsp; $\left| H_{d}(e^{j2 \pi f}) \right|=\left| \frac{sin(2\pi f/2)}{sin(2\pi fD/2)}\right|^M$
2. Compute the impulse responses:
&emsp; &emsp; $h_d[n]=\frac{1}{2\pi}\int_{0}^{2\pi}H_d(e^{jw})e^{jwn}dw$

3. FIR Approximation via Kaiser Windowing: $h[n]=h_d[n]w[n]$
&emsp; &emsp; $h[n] = \begin{cases} h_d[n] & 0\leq n\leq N \\ 0 & otherwise \end{cases}$
&emsp; &emsp; $w[n] = \begin{cases} 
          \frac{I_0(\beta\sqrt{1-(n-\alpha)^2/\alpha^2})}{I_0(\beta)} & 0\leq n\leq N \\
          0 & otherwise
       \end{cases}$
 4. Quantization 
 5. Canonical Sign Digit Representation Conversion  


This design process it achieved through a python script in python script that generates the compensation FIR filter verilog with inputted filter parameters:

| Parameter | Description |
|--------|-------------|
| $f_{in}$ | Input Frequency for CIC  |
| $f_{out}$ | Output Frequency for CIC |
| $f_{N}$ | Nyquist Frequency |
| $R$ | Decimation Rate |
| $N_{CIC}$ | Number of CIC Stages |
| $N_{TAPS}$ | Number of taps |
| $f_{p}$ | Cutoff Frequency/Pass-band Frequency   |
| $N_{BITS}$ | Bit width of Tap Coefficients | 
| $\beta$ | Kaiser Window Shape Parameter |
| $IW$ | Input Bit Width |
