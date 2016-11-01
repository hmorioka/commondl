# commondl

Matlab implementation of the common dictionary leaning algorithm from Morioka et al., NeuroImage 2015.

Morioka H., Kanemura A., Hirayama J., Shikauchi M., Ogawa T., Ikeda S., Kawanabe M., Ishii S. Learning a common dictionary for subject-transfer decoding with resting calibration. Neuroimage, 111:167–178 (2015).

Version 1.0, July 1 2015  
Author: Hiroshi Morioka.  
License: Apache License, Version 2.0  


Common dictionary learning programs (./commonDL):  
  commonDL        - main algorithm of the common dictionary learning  
  runwiseSC       - run-wise sparse coding  
  schedSampling   - needed by commonDL  

Demo programs:  
  demo                     - main demo program  
  set_path                 - set up path  
  generate_artificial_data - generate artificial signals used in simulations  
  showmatgrid              - for plotting bases  
