# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:37:11 2025



StateScript Built-Ins and Helpers:
    
    These methods are largely intended to call the built-in methods in 
    StateScript. 
    
    send_message(command:str, log_message:bool = True) : 
        formats and sends a command to StateScript, optionally logging 
        the command in the python log. This is not itself a StateScript 
        built-in, but it is what all the other built-ins are based around.
        Example:
            >>> send_message('portout[3]=flip')
            SCQTMESSAGE: portout[3]=flip;
        Note how this method adds the prefix "SCQTMESSAGE: ", and the 
        suffix ";", and then prints the message to the console (where 
        the python_observer script will pass the message to StateScript)
    
    disp(message:str) : 
        displays a message in StateScript (does check for/remove characters 
        which will cause errors in StateScript). Executes:
            disp('<message>')
    
    disp_var(var_name:str) : 
        displays the value of a variable in StateScript. Executes:
            disp(<var_name>);
    
    portout(port:int, state:int) : 
        sets the output of the specified port. Executes: 
            "portout[<port>]=<state>;"
    
    flip(port:int) : 
        flips the output value of the specified port by calling 
        "portout[<port>]=flip;" directly rather than by computing 
        the value to set the port to first and then calling 
        portout(port, 1 - curr_val) in python. Executes: 
            "portout[<port>] = flip;"
    
    trigger(function:int) : 
        Triggers the specified function in StateScript. If 
        StateScriptInterface.functions has been set, you can also trigger 
        functions by name. E.g.:
            >>> ssi.functions['reset'] = 4
            >>> ssi.trigger('reset')
            SCQTMESSAGE: trigger(4);
        Executes:
            "trigger(<functions[<function>]>);" if function in functions else "trigger(<function>);"
    
    set_var(var_name:str, value:int) : 
        Sets the value of the StateScript variable to the specified 
        value (note that StateScript currently only supports integers).
        Executes: 
            "<var_name> = <value>;"
    
    increment(var_name:str, increment_by:int = 1) : 
        Increments the value of the StateScript variable by 
        the specified amount.
        Executes: 
            "<var_name> = <var_name> + <increment_by>"
        (Does know how to subtract if increment_by < 0)
    
    decrement(var_name:str, decrement_by:int = 1) : 
        Decrements the value of the StateScript variable by 
        the specified amount.
        Executes: 
            "<var_name> = <var_name> - <decrement_by>"
        (Does know how to add if decrement_by < 0)
    
    random(var_name:str, high:int) : 
        set the value of the StateScript variable to a number which 
        StateScript generates randomly on the interval [0, high].
        Executes: 
            "<var_name> = random(<high>);"
    
    sound(filename:str) : 
        Start playing the sound file at the specified filename, or 
        stop sound playback if filename == 'stop'. Executes:
            "sound(stop);" if filename == 'stop' else "sound('<filename>');"
    
    volume(value:int) : 
        Sets the volume for sound playback. Executes: 
            "volume(<value>);"
    
    updates(state:int, port:int = None) : 
        Turn on/off DIO auto state updates (for the specified port). state 
        must either be 'on' or 'off'. Executes:
            "updates <state>;" if port == None else "updates <state> <port>;"
    
    clock(var_name:str, reset:bool = False) : 
        store the current clock time in the specified variable, restting 
        the clock if reset == True. Executes: 
            "<var_name> = clock(reset);" if var_name == 'reset' else "<var_name> = clock();"
    
    thresh(aio:int, threshold:int) : 
        Set the detection threshold for the specified analog input (aio) 
        to <threshold> mV. Executes: 
            "thresh on <aio> <threshold>;"

@author: Violet
"""

import os
import re
import time
import platform
from collections import defaultdict

"""Constants"""
filepath_separator = '\\' if platform.system() == 'Windows' else '/' #  / works for both linux and mac

"""
Basic Interface Commands

These commands largely replicate the StateScript built-in functions as 
found at https://docs.spikegadgets.com/en/latest/basic/StateScript.html


"""

# Send a Message to StateScript (base command wrapper)
def send_message(command:str, log_message:bool = True):
    """send a command to StateScript (print(f"SCQTMESSAGE: {command};\n"))
    
    NOTE: this method is unprotected, please consider using these methods instead: 
        - disp/disp_var
        - portout
        - (flip)
        - trigger
        - (set_var)
        - random
        - sound
        - volume
        - updates
        - clock
        - thresh
    With the exceptions of flip() and set_var(), these methods mimic the 
    StateScript built-in methods (disp_var mimics disp(variable_name)).
    set_var() isn't a StateScript built-in, but it allows you to easily 
    set the values of variables in StateScript. And flip() runs the command 
    portout[port]=flip; directly, rather than portout(port, 'flip') which 
    pre-computes the value to set the port to (i.e. it either ends up 
    executing the command portout[port]=1; or portout[port]=0; depending 
    on the current state of the port).
    """
    command = f"SCQTMESSAGE: {command};"
    if config.auto_log_commands and log_message: log(command)
    print(command) #  automatically adds '\n' to the end of the command

def disp(string:str, log_message:bool = True):
    """display a message in the StateScript terminal"""
    string = _sanitize_message(string)
    if config.auto_log_commands and log_message: log(string)
    send_message(f"disp('{string}')", False)

def disp_var(var_name:str, log_message:bool = True):
    """display the value of a StateScript variable"""
    send_message(f"disp({_sanitize_message(var_name)})", log_message)

def portout(port:int, value:int, log_message:bool = True):
    """set the output state of a port (state in {0, 1, 'flip'})"""
    # Check the Inputs
    if config.enable_assertions:
        assert _is_int_str(port)
        assert _is_int_str(value) or value == 'flip'
    
    # Send the Message to StateScript
    send_message(f"portout[{port}] = {value}", log_message)

def flip(port:int, log_message:bool = True): # this is actually a custom command
    """flip the output of a port (same function as StateScriptInterface.toggle)"""
    if config.enable_assertions:
        assert _is_int_str(port)
    send_message(f"portout[{port}] = flip", log_message)
toggle = flip # alias of flip

functions = {} # a dictionary which maps names for StateScript functions to StateScript function numbers
def trigger(func:int, log_message:bool = True):
    """trigger a StateScript function
    
    gets the function number from the StateScriptInterface.functions 
    dictionary when possible. (i.e. trigger(functions[func]) if func in functions else trigger(func))
    """
    if config.enable_assertions:
        assert _is_int_str(func) or (type(func) in (str, int, tuple) and func in functions)
    func = functions[func] if func in functions else func
    if config.enable_assertions:
        assert _is_int_str(func)
    send_message(f"trigger({func})", log_message)

def set_var(var_name:str, value:int, log_message:bool = True):
    """Set the value of a variable in StateScript
    
    Note that StateScript currently only supports integer variable types
    """
    if config.enable_assertions:
        assert type(var_name) == str
        assert _is_int_str(value) #  TODO: remove this if/when SpikeGadgets adds non-integer variable types to StateScript
    send_message(f"{var_name} = {value}", log_message)

def increment(var_name:str, increment_by:int = 1, log_message:bool = True):
    """Increment the StateScript Variable by <increment_by>"""
    if config.enable_assertions:
        assert type(var_name) == str
        assert _is_int_str(increment_by)
    increment_by = int(increment_by)
    operator = '+' if increment_by >= 0 else '-'
    send_message(f"{var_name} = {var_name} {operator} {abs(increment_by)}")

def decrement(var_name:str, decrement_by:int = 1, log_message:bool = True):
    """Decrement the StateScript Variable by <decrement_by>"""
    if config.enable_assertions:
        assert type(var_name) == str
        assert _is_int_str(decrement_by)
    decrement_by = int(decrement_by)
    operator = '-' if decrement_by >= 0 else '+'
    send_message(f"{var_name} = {var_name} {operator} {abs(decrement_by)}")

def random(var_name:str, high:int, log_message:bool = True):
    """generates a random integer from 0-high (inclusive?) in StateScript, and stores the result in the specified value"""
    if config.enable_assertions:
        assert type(var_name) == str
        assert _is_int_str(high)
        send_message(f"{var_name} = random({high})", log_message)

def sound(filename:str, log_message:bool = True):
    """play the specified sound file (or 'stop' to stop audio playback before it's done)"""
    if config.enable_assertions:
        assert type(filename) == str
    if filename == 'stop':
        send_message("sound(stop)", log_message)
    else:
        send_message(f"sound('{filename}')", log_message)

def volume(value:int, log_message:bool = True):
    """set the current audio volume"""
    if config.enable_assertions:
        assert _is_int_str(value)
    send_message(f"volume({value})", log_message)

def updates(state:str, port:int = None, log_message:bool = True):
    """turn on/off DIO auto state updates (for the specified port). state in {'on', 'off'}"""
    if config.enable_assertions:
        assert type(state) == str and state in ('on', 'off')
        assert port == None or _is_int_str(port)
    if state == 'on':
        send_message("updates on", log_message)
    elif config.enable_assertions or state == 'off':
        if port == None:
            send_message("updates off", log_message)
        else:
            send_message(f"updates off {port}", log_message)
    else:
        raise ValueError(f"<state> must either by 'on' or 'off', not '{state}'")

def clock(var_name, reset:bool = False, log_message:bool = True):
    """store the current clock time in the specified variable (resetting the clock if reset == True)"""
    if config.enable_assertions:
        assert type(var_name) == str
    if reset:
        send_message(f"{var_name} = clock(reset)", log_message)
    else:
        send_message(f"{var_name} = clock()", log_message)

def thresh(aio:int, threshold:int, log_message:bool = True):
    """set the detection threshold for the specified analog input (aio) to <threshold> mV"""
    if config.enable_assertions:
        assert _is_int_str(aio)
        assert _is_int_str(threshold)
    send_message(f"thresh on {aio} {threshold}", log_message)


"""
Track the Input/Output State of the DIO Ports
"""

class state:
    """
    holds the most up-to-date information on the states of all 
    digial inputs/outputs as bitmasks. Only works if using the 
    StateScriptInterface callback() functions (or another 
    callback function built on StateScriptInterface._callback()). 
    Alternatively, you can update the state by hand every time a 
    line with 3 integers appears in statescript by calling the 
    StateScriptInterface.state.update() function yourself.
    """
    
    # the input/output states
    inputs = 0
    outputs = 0
    
    # the change from the previous input/output states
    d_inputs = 0
    d_outputs = 0
    
    # Port Number -> Port Objects
    ports = {}
    
    def update(inputs = None, outputs = None):
        # Get the input/output bitmasks
        inputs = _null_coalescing(inputs, state.inputs)
        outputs = _null_coalescing(outputs, state.outputs)
        
        # Compute the Change from the Previous State
        state.d_inputs = inputs ^ state.inputs
        state.d_outputs = outputs ^ state.outputs
        
        # Store the Input/Output States
        state.inputs = inputs
        state.outputs = outputs
        
        # Update All the Port Objects
        for port_group in config.port_groups.values():
            for port in port_group:
                if port in state.ports:
                    state.ports[port]._update_state()
    
    def port_to_bitmask(port:int):
        return 1 << (port - 1)
    
    def get_input_state(port:int):
        return int(bool(state.inputs & state.port_to_bitmask(port)))
    
    def get_output_state(port:int):
        return int(bool(state.outputs & state.port_to_bitmask(port)))
    
    def set_output(port:int, value:int, log_message:bool = True):
        value = _sanitize_state(value)
        if value == 2: value = 1 - state.get_output_state(port)
        portout(port, value, log_message)


"""
Port Config
"""

class Port:
    """An Object for Managing the Input/Output of a Single DIO Port
    
    Main Methods
    ------------
    set_output(), on(), off(), flip() - update the output state of the port (immediately 
        if delay_outputs == False, or the next time update_output() is called if 
        delay_outputs == True).
    
    
    Attributes
    ----------
    port : int
        The port number (as recornized by the ECU)
    port_group : str
        The group of objects the port belongs to (E.g. 'reward_wells')
    index : int
        The index of the Port object in the list of Port objects which 
        share the same group (the index is not unique between all ports 
        unless all Port objects share the same group)
    bitmask : int
        A filter for interpreting the input/output state of the port from 
        state updates from StateScript. (i.e. when StateScript outputs 3 
        integers, the first integer is a timestamp (like usual), the second 
        integer is a bitmask representing which inputs are high, and the 
        third integer is a bitmask representing which outputs are high).
        The value of this bitmask should always be 2**(port - 1), or 
        equivalently: 1 << (port - 1). DO NOT EDIT THIS VALUE
    input_val : int
        The current input value of the port (only udpates itself when the 
        callback function is built around StateScriptInterface._callback)
    output_val : int
        The current output value of the port (only udpates itself when the 
        callback function is built around StateScriptInterface._callback. 
        Otherwise it assumes the output is initialized to 0, and then tracks 
        when the value is updated in Python - this can be broken if the 
        output value is set directly from StateScript)
    next_output_val : int
        The value to update the output to the next time update_output() is 
        called. (only used if delay_outputs == True)
    delay_outputs : bool
        When true, methods like set_output()/on()/off()/flip() do not 
        call portout() right away. Instead they update next_output_val, 
        and then the ECU output will be updated when update_output() is 
        called. (this is useful when you want to calculate what the next 
        output should be on one callback trigger, and then update the 
        output state on a different callback trigger). Default is False.
    
    """
    def __init__(self, port:int, group:str = 'misc', delayed_outputs:bool = False, dio_updates:bool = True):
        """Create an Object to Manage the Input/Output State of a DIO Port

        Parameters
        ----------
        port : int
            The port number (as interpreted by the ECU).
        group : str, optional (but recommended)
            The group of objects the port is a member of
            (e.g. 'Well', or 'Pump'). Default = 'misc'

        """
        
        # Set the Attributes, and Store the Port in the Ports List
        self.port = port
        self.port_group = group
        self.index = config.get_port_index(self.port) if config.has_port(port) else config.add_port_to_group(self.port, self.port_group)
        self.bitmask = 1 << (self.port - 1)
        self.input_val = 0
        self.output_val = 0
        self.next_output_val = None # 0/1/2/None = (low/high/flip/no-update)
        self.delay_outputs = delayed_outputs
        self.dio_updates = dio_updates
        
        # Store a Pointer to the Port Object
        state.ports[self.port] = self
        
        # Turn Off DIO Updates
        if not self.dio_updates: self.updates_off()
        
    
    def __hash__(self) -> int:
        """Returns A Unique Identifier for the Port (the port number)"""
        return self.port #  the indices are by group, so they're not unique, but the port numbers are
    
    # a string representation of a port
    def __repr__(self) -> str:
        """Creates a String Representation of the Port Object"""
        if type(self.port_group) == str:
            return f"<{self.port_group} Port {self.index}>"
        elif type(self.port_group) == int:
            return f"<Group {self.port_group} Port {self.index}>"
        else:
            return f"<Port {self.port}>"
    
    # check if 2 pumps are equal
    def __eq__(self, other) -> bool:
        """Checks if 2 Ports Are Equal"""
        return type(other) == type(self) and self.port == other.port
    
    def _update_state(self):
        """Update the Input/Output Val (called when _callback() recieved a state update from the ECU)"""
        self.get_input_from_state()
        self.get_output_from_state()
    
    def get_input_from_state(self) -> int:
        """Get the Current Input Value of the Port (0/1)
        
        Requires that StateScriptInterface.state is being maintained (which happens 
        automatically when using callbacks based on StateScriptInterface._callback
        """
        self.input_val = int(bool(self.bitmask & state.inputs))
        return self.input_val
    
    def get_output_from_state(self) -> int:
        """Get the Current Output State of the Port (0/1)
        
        Requires that StateScriptInterface.state is being maintained (which happens 
        automatically when using callbacks based on StateScriptInterface._callback
        """
        self.output_val = int(bool(self.bitmask & state.outputs))
        return self.output_val
        
    
    def get_input(self) -> int:
        """Get the Current Input Value of the Port (0/1)
        
        Requires that StateScriptInterface.state is being maintained (which happens 
        automatically when using callbacks based on StateScriptInterface._callback
        """
        return self.input_val
    
    def get_output(self) -> int:
        """Get the Current Output State of the Port (0/1)
        
        Requires that all outputs start off, and that outputs are only updated 
        from python via this object (otherwise the state stored in python and 
        the actual state could be mismatched, leading to errors)
        """
        return self.output_val
    
    def set_output(self, state:int, set_immediate:bool = False, log_message = True) -> None:
        """Set the Output of the Port. Roughly Equivalent to 'portout[<port>]=<state>;'"""
        state = _get_state(state, self.output_val)
        if not config.blit_portouts or state != self.output_val or state != self.next_output_val:
            if self.delay_outputs and not set_immediate:
                self.next_output_val = state
            else:
                self.output_val = state
                self.next_output_val = None
                portout(self.port, state, log_message)
    
    def on(self, set_immediate:bool = False, log_message = True) -> None:
        """Set the Output of the Port to High. Equivalent to 'portout[<port>]=1;'"""
        self.set_output(1, set_immediate, log_message)
    high = on #  alias of Port.on
    
    def off(self, set_immediate:bool = False, log_message = True) -> None:
        """Set the Output of the Port to Low. Equivalent to 'portout[<port>]=0;'"""
        self.set_output(0, set_immediate, log_message)
    low = off #  alias of Port.off
    
    def flip(self, set_immediate:bool = False, log_message = True) -> None:
        """flip the Output of the Port. Equivalent to 'portout[<port>]=flip;'"""
        if self.delay_outputs and not set_immediate:
            self.next_output_val = 1 - self.output_val
        else:
            self.output_val = 1 - self.output_val
            self.clear_next_output()
            flip(self.port, log_message)
    toggle = flip #  alias of Port.flip
    
    def update_output(self, log_message:bool = True) -> None:
        """Set the Output of the Port to self.next_output_val (if next_output_val != None)"""
        if self.next_output_val != None:
            self.set_output(self.next_output_val, True, log_message)
            self.clear_next_output()
    
    def clear_next_output(self):
        """Clear the next_output Value"""
        self.next_output_val = None
    
    def updates_off(self, log_message:bool = True) -> None:
        """Turn StateScript DIO Updates Off for this Port"""
        self.dio_updates = False
        updates('off', self.port, log_message)
    
    def updates_on(self, log_message:bool = True) -> None:
        """Turn StateScript DIO Updates On for this Port"""
        self.dio_updates = True
        updates('on', self.port, log_message)
    
    def toggle_updates(self, log_message:bool = True) -> bool:
        """Toggle StateScript DIO Updates for this Port, Returning if Updates are On"""
        if self.dio_updates:
            self.updates_off(log_message)
        else:
            self.updates_on(log_message)
        return self.dio_updates

class config:
    port_groups = {}
    _port_reverse_mapping = {}
    command_handlers = {}
    command_regex = '^\d+ '
    stats_function = None
    
    blit_portouts = True # only do portout commands which will actually change the ECU output state
    
    rewarded_command = 'UP'
    
    state_callback_handler = None
    
    state_regex = re.compile('^\d+ \d+ \d+$')
    state_parser = lambda state: tuple(map(int, state.split(' ')))
    
    input_exclusions = defaultdict(list)
    output_exclusions = defaultdict(list)
    
    auto_log_commands = True
    user_commands_enabled = True
    
    enable_assertions = True    #  enables data checking in commands
    verbose = False             #  enables extra print statements
    
    def has_port(port:int):
        return port in config._port_reverse_mapping
    
    def _get_port_list(ports) -> list:
        if type(ports) == list:
            return ports
        elif ports == None or ports == 'all':
            return list(config._port_reverse_mapping.keys())
        elif type(ports) in (str, int, tuple) and ports in config.port_groups:
            return config.port_groups[ports]
        else:
            raise ValueError(f"Unrecognized ports signifier: '{ports}'")
    
    def add_port_to_group(port:int, group):
        if group not in config.port_groups:
            config.port_groups[group] = []
        index = len(config.port_groups[group])
        config._port_reverse_mapping[port] = index
        config.port_groups[group].append(port)
        return index
    
    def add_port_group(ports:list, group):
        for port in ports:
            config.add_port_to_group(port, group)
    
    def add_command(command:str, callback):
        config.command_handlers[command] = callback
        config.set_command_regex()
    
    def add_commands(command_dict:dict):
        for key, func in command_dict.items():
            config.add_command(key, func)
    
    def set_command_regex(regex:str = None):
        """Set the Regex Used for Itentifying Valid Commands
        
        If 'regex' is None, a standard format will be used 
        based off of the commands in config.command_handlers:
            ^\d+ (COMMAND 1|COMMAND 2|...|COMMAND N)( \d+)?$
        Which equates to:
            <start of string><timestamp> <valid opcode>[ <port number (optional)>]<end of sring>
        """
        if regex:
            config.command_regex = re.compile(regex) if type(regex) == str else regex
        else:
            config.command_regex = re.compile(r"^\d+ (" + '|'.join(config.command_handlers.keys()) + ')( \d+)?$') # "<Timestamp> <Opcode>( <value>)"
    
    def configure_ports(port_groups:dict):
        """Configure the Active DIO Ports. port_groups = dict(group_name -> list_of_port_numbers)"""
        for group, port_list in port_groups:
            config.add_port_group(port_list, group)
    
    def get_port(group, index:int):
        return config.port_groups[group][index]
    
    def get_port_index(port:int):
        return config._port_reverse_mapping[port]
    
    def forbid_simultaneous_inputs(ports:list, name:str = None):
        """assert that no 2 Ports in the list can have active inputs at the same time.
        
        this only works when using callback functions based on ssi._callback()
        """
        if name == None and type(ports) == str: name = ports
        ports = config._get_port_list(ports)
        
        names = set()
        bitmask = 0
        for port in ports:
            if type(port) == int:
                bitmask |= 1 << (port - 1)
            elif isinstance(port, Port):
                bitmask |= port.bitmask
                names.add(port.port_group)
            else:
                print(f'Warning: unable to get a bitmask for objects of type {type(port)}')
        
        name = _null_coalescing(name, '|'.join(names))
        
        exclusion_list = config.input_exclusions[name]
        if bitmask not in exclusion_list:
            exclusion_list.append(bitmask)
    
    def forbid_simultaneous_outputs(ports:list, name:str = None):
        """assert that no 2 Ports in the list can have active outputs at the same time.
        
        this only works when using callback functions based on ssi._callback()
        """
        if name == None and type(ports) == str: name = ports
        ports = config._get_port_list(ports)
        
        names = set()
        bitmask = 0
        for port in ports:
            if type(port) == int:
                bitmask |= 1 << (port - 1)
            elif isinstance(port, Port):
                bitmask |= port.bitmask
                names.add(port.port_group)
            else:
                print(f'Warning: unable to get a bitmask for objects of type {type(port)}')
        
        name = _null_coalescing(name, '|'.join(names))
        
        exclusion_list = config.output_exclusions[name]
        if bitmask not in exclusion_list:
            exclusion_list.append(bitmask)
    
    def configure_dio_updates(ports_on:list = None, log_messages:bool = True):
        """Turn Off DIO Updates, then Turn On DIO Updates for the Ports in <ports_on>.
        
        If ports_on == None then all active ports will have updates turned on"""
        updates('off', None, log_messages)
        ports_on = _null_coalescing(ports_on, config._port_reverse_mapping.values())
        for port in ports:
            if isinstance(port, Port):
                port = port.port
            updates('on', port, log_messages)

# Reduce the Ammount of Typing
ports = config.port_groups
add_command = config.add_command
add_commands = config.add_commands

# Update the Outputs of All the Ports
def update_ports(ports:str = None):
    """Update the Outputs of All the Specified Ports
    
    <ports> must either be the name of a port group (usually a string), a 
    list/iterable of Port objects, or None (which )
    """
    
    # Get the Port List
    if ports == None:
        ports = config._port_reverse_mapping.keys()
    elif type(ports) in (str, int, tuple) and ports in config.port_groups:
        ports = config.port_groups[ports]
    elif type(ports) != list:
        if hasattr(ports, '__iter__'): 
            ports = list(ports)
        else:
            raise ValueError(f"Unrecognized Port-List Specifier: '{ports}'")
    
    # Update the Ports
    for port in ports:
        port.update_output()

"""
Callback Command Setup
"""

# Command Decorator
def command(command_handler):
    """A Decorator which Lets StateScriptInterface know a Function is a Command Handler
    
    The name of the function should mirror the command which will call the function. E.g.
        
        @StateScriptInterfact.command
        def UP(well):
            <do something>
            
    will map the string 'UP' to the function UP, so that when the callback 
    function recieves a message like "<timestamp> UP <port>" it will find 
    the Well object associated with the port triggered, and then pass that 
    Well object to the function UP:
        
        well_index = config.get_input_index(int(<port>))
        well = Well.wells[well_index]
        UP(well)
    
    
    
    Note that the name of the function must match the command string 
    exactly (i.e. it's case sensative, and cannot contain extra characters).
    So for instance:
        
        @ssi.command
        def up(well):
            <do something>
    
    will only be called if the callback function sees the command "up", 
    not "Up", "UP", "uP", or any other combination of characters besides 
    "up".
    
    
    
    Also note that for the built-in callback function, all command handlers 
    are passed a parameter called "well", which is either the reward well 
    which was just triggered, or None if no reward well is 
    
    
    
    Lastly, it is possible to have the key not match the function name 
    by doing:
        
        @ssi.command('<name>')
        def func(well):
            <do something>
    
    But for simplicity, I don't really recommend doing this
    
    """
    
    # a helper method for adding commands (just used by this decorator)
    def _add_command_local(key:str, handler):
        # add the command to the dictionary
        config.add_command(key, handler)
        
        # print that the command was added
        if config.verbose: print(f"command handler added with key: '{key}'")
        
        # return the function (a necessary step for decorators)
        return handler
    
    # check what was provided for 'command_handler'
    if hasattr(command_handler, '__call__'):
        # command_handler was a function, so add it to the dictionary directly
        return _add_command_local(str(command_handler).split(' ')[1], command_handler)
    elif type(command_handler) == str:
        # command_handler was a string, so return a decorator which will then map a function to the key provided
        return lambda func: _add_command_local(command_handler, func)
    else:
        # unrecognized type for command_handler
        disp('Python Error: python is not running') # get the user's attention
        raise TypeError('command_handler must either be a function or a string') # shut down the python script

def command_is_valid(command:str):
    """Convenience Method for Checking if a Command is Valid (uses config.command_regex)"""
    return config.command_regex.match(command) != None

def print_stats():
    """Trigger the StateScript Stats Function (if specified)"""
    if config.stats_function != None: 
        trigger(config.stats_function)
    elif 'stats' in functions:
        trigger('stats')

"""
Logger
"""

# Get a DateTime Stamp
def get_timestamp(format_str = "%Y%m%d_%H%M%S") -> str:
    """Get a Timestamp Formatted Using the Given Format String
    
    See doccumentation for the time.py library for how to 
    construct a format string:
        https://docs.python.org/3/library/time.html for 
    """
    return time.strftime(format_str.replace("%f", str(int(round(time.time()%1 * 10**6)))))

# Data Logging Static Class
class Logger:
    
    # Default Timestamp Format String: "<hours>:<minuts>:<seconds>.<microseconds>"
    default_timestamp_format_str = "%H:%M:%S.%f"
    
    def __init__(self, rat_name:str = None, folder:str = None, timestamp_format_str = default_timestamp_format_str):
        # Log Info
        self.filepath = None
        self.filename = None
        self.folder = None
        self.is_open = False
        self.timestamp_format_str = None
        
        # Open the Log with Default Params
        if rat_name and folder:
            self.open(Logger.get_default_filepath(rat_name, folder, timestamp_format_str))
    
    # Get a Timestamp
    def _get_timestamp(self) -> str:
        return get_timestamp(self.timestamp_format_str) if self.timestamp_format_str else get_timestamp()
    
    # Get the Default Filename
    def get_default_filepath(rat_name:str, folder:str, suffix = '.log') -> str:
        date = get_timestamp("%Y%m%d")
        prefix = f"{date}_{rat_name}"
        files = [fname for fname in os.listdir(folder) if fname.startswith(prefix) and fname.endswith(suffix)]
        rec_number = len(files) + 1
        filename = f"{prefix}_{rec_number}{suffix}"
        while filename in files:
            rec_number += 1
            filename = f"{prefix}_{rec_number}{suffix}"
        return f"{folder}{filepath_separator}{filename}"
    
    # Open a Log File
    def open(self, filepath:str, timestamp_format_str = default_timestamp_format_str):
        """Initialize a Log File at the Given Filepath"""
        # Save the Filepath/Filename/Folder
        self.filepath = filepath
        info = filepath.split(filepath_separator)
        self.filename = info[-1]
        self.folder = filepath_separator.join(info[:-1])
        self.timestamp_format_str = timestamp_format_str
        
        # check for filename collisions (fixing collisions by copying the old file)
        if os.path.isfile(self.filepath):
            fname = self.filename
            suffix = fname.split('.')[-1]
            fname = fname[:-1 - len(suffix)]
            copy_path = self.filename
            count = 0
            while copy_path in os.listdir(self.folder):
                count += 1
                copy_path = f"{fname}_{count}.{suffix}"
            utils.copy_file(self.filepath, filepath_separator.join((self.folder, copy_path)))
            disp(f"warning: a file named {self.filename} already exists in the specified folder. This file has been renamed {copy_path}")
        
        # write a litle header
        timestamp = self._get_timestamp()
        message = f"[{timestamp}] log file '{self.filename}' opened"
        with open(self.filepath, 'w') as file:
            file.write(message + '\n')
            file.write('='*len(message) + '\n')
            file.close()
        
        # flag that the log is Open
        disp(message.replace("'", ''), False)
        self.is_open = True
    
    # Close a Log File
    def close(self):
        """Close the Log File that is Currently Open"""
        if self.is_open and self.filepath:
            # write a little footer
            timestamp = self._get_timestamp()
            message = f"[{timestamp}] log file '{self.filename}' closed"
            with open(self.filepath, 'a') as file:
                file.write('='*len(message) + '\n')
                file.write(message)
                file.close()
            
            # flag that the log is closed
            self.is_open = False
            disp(message.replace("'", ''), False)
    
    # Add a Line-Break
    def add_line_break(self, length:int = 64, char:str = '='):
        """Adds <char>*<length> to the log file, without a timestamp"""
        self.add_line_without_timestamp(char * length)
    
    # Add to the Log File
    def log(self, line:str):
        """Append a Line to the Log File"""
        if self.is_open and self.filepath:
            timestamp = self._get_timestamp()
            with open(self.filepath, 'a') as file:
                file.write(f"[{timestamp}] {line}\n")
                file.close()
        elif config.verbose:
            disp('failed to log message, see python console')
            print(f"there is no open log file to record the message: '{line}'")
    
    # Add a Line to the Log Without a Timestampe
    def add_line_without_timestamp(self, line:str):
        """Append a Line to the Log File Without a Timestamp"""
        if self.is_open and self.filepath:
            with open(self.filepath, 'a') as file:
                file.write(line)
                file.write('\n')
                file.close()
        elif config.verbose:
            disp('failed to log message, see python console')
            print(f"there is no open log file to record the message: '{line}'")

# Initialize the Main Log
logger = Logger()

# logging shortcuts
close_log = logger.close
log = logger.log

# Open a Log
def open_log(rat_name:str, folder:str = None, timestamp_format_str = Logger.default_timestamp_format_str):
    """Log-Opening Convenience Method
    
    Calls: 
        logger.open(rat_name, timestamp_format_str) if folder == None
    Otherwise calls:
        logger.open(Logger.get_default_filepath(rat_name, folder), timestamp_format_str)
    """
    if folder == None:
        logger.open(rat_name, timestamp_format_str)
    else:
        logger.open(Logger.get_default_filepath(rat_name, folder), timestamp_format_str)


"""
Helpers
"""

# Null Coalescing Operator
def _null_coalescing(value, default):
    """returns <value> if <value>!=None else <default>"""
    return value if value != None else default

def _is_int_str(value) -> bool:
    return type(value) == int or (type(value) == str and value.isdigit())

def _sanitize_state(state:int) -> int:
    """Helper Function for Determining Which Values for 'state' Should Eval to 1 or 0 (or 2)"""
    if state in (1, True):
        return 1
    elif state in (0, False):
        return 0
    elif state == 2:
        return 2
    elif type(state) == str:
        state = state.lower()
        if state in ('flip', 'toggle'):
            return 2
        elif state in ('high', 'true', 'on'):
            return 1
        elif state in ('low', 'false', 'off'):
            return 0
    
    # If we've reached this point the state hasn't been recognized
    raise ValueError(f"'state' must either be 0/False/'off'/'low', 1/True/'on'/'high', or 2/'flip'/'toggle', not '{state}'")

def _get_state(new_state:int, current_state:int) -> (int, bool):
    """Helper Function for Computing State Changes"""
    new_state = _sanitize_state(new_state)
    current_state = _sanitize_state(current_state)
    return 1 - current_state if new_state == 2 else new_state

# Sanitize a Message for the Disp Command
def _sanitize_message(msg:str) -> str:
    forbidden_chars = {"'", ')', '%'} # single quotes aren't necessarily forbidden, but they do behave weird lol
    if forbidden_chars.intersection(msg):
        print(f"warning: these characters cannot be printed in StateScript, and have been removed from the message: {forbidden_chars}")
        for c in forbidden_chars:
            msg = msg.replace(c, '')
    return msg


"""
Utils
"""

class utils:
    def readlines(filepath:str) -> list:
        """read a text file, storing the lines in a list (removing trailing \n characters)"""
        with open(filepath, 'r') as f:
            lines = [l[:-1] if l.endswith('\n') else l for l in f.readlines()]
            f.close()
        return lines
    
    def writelines(filepath:str, lines:list) -> None:
        """write the given lines to a text file at <filepath> (adding \n characters to the end of each line)"""
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
            f.close()
    
    def copy_file(src:str, dst:str) -> None:
        utils.writelines(dst, utils.readlines(src))
    
    def copy_training_logs(fname:str, names:list):
        lines = None
        with open(fname, 'r') as f:
            lines = f.readlines()
            f.close()
        
        directory = os.path.dirname(fname)
        suffix = fname.split('_')[-1]
        
        for name in names:
            with open(f"{directory}{os.path.sep}{name}_{suffix}", 'w') as f:
                f.writelines(lines)
                f.close()
    
    def copy_python(fname:str, names:list):
        lines = None
        with open(fname, 'r') as f:
            lines = f.readlines()
            f.close()
        
        directory = os.path.dirname(fname)
        
        for name in names:
            with open(f"{directory}{os.path.sep}6arm_{name}.py", 'w') as f:
                f.writelines(lines)
                f.close()




"""
CALLBACK HELPERS
"""

# A Function Used to Parse Commands (override if necessary) - use in conjunction with config.command_regex
def command_parser(line:str) -> (int, str, int):
    """Default Command Parsing Function (expects '<timestamp, int> <opcode, str> <port number, int, optional>')"""
    line = line.split(' ')
    t = int(line[0])
    op = line[1]
    val = int(line[2]) if len(line) == 3 else None
    return t, op, val


# A Callback Function Helper
def parse_command(line:str) -> (bool, int, str, int):
    """
    Parse a Command from StateScript to Python
    
    
    """
    if config.command_regex.match(line):
        # Log Commands
        log(line)
        
        # Parse the Line - Default Format: "<timestamp> <opcode> (<optional port>)"
        data = command_parser(line)
        
        # Return that the Command was Matched
        return True, *data
    elif config.state_regex.match(line):
        # Log State Updates
        log(line)
        
        # Parse the Line
        t, inputs, outputs = config.state_parser(line)
        
        # Update the Stored State
        state.update(inputs, outputs)
        
        # Check for Error States
        for state_bitmask, exclusions in zip((inputs, outputs), (config.input_exclusions, config.output_exclusions)):
            for name, exclusion_bitmasks in exclusions.items():
                for bitmask in exclusion_bitmasks:
                    # Count the Number of 1's
                    x = state_bitmask & bitmask
                    count = 0
                    while x:
                        # Update the Count
                        count += x&1
                        
                        # Check if an Exclusion was Found
                        if count > 1:
                            disp(f'FORBIDDEN DIO COMBINATION ENCOUNTERED: <{name}>. PLEASE CHECK ENVIRONMENT HARDWARE')
                            log(f'{line} , <{name}> exclusion bitmask: {bitmask}')
                            break
                        
                        # Move On to the Next Bit
                        x >>= 1
        
        # Check for a Callback Handler
        if hasattr(config.state_callback_handler, '__call__'):
            config.state_callback_handler(t, inputs, outputs)
    
    # Return that no Command was Matched
    return False, None, None, None
