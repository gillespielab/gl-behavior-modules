# gl-behavior-modules
Modules for Running Behavior with Trodes/StateScript, and for Analyzing Behavior Data

### StateScriptInterface
In developing both this environment and other environments in the past, one common source of errors has been StateScript. Communication between python and StateScript in particular has caused numerous bugs. To address this, a StateScriptInterface was developed in python which defines wrappers for all of the StateScript built-in functions, as well as a class for managing DIO ports, and a static class for tracking input/output states as reported by the ECU. Additionally, StateScriptInterface provides a \_\_callback\_\_ function, which sends data to the appropriate handler using the format: \"\<timestamp\> \<opcode\> \[\<value/>\]\".
