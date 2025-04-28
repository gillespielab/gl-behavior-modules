# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:34:28 2025

@author: Violet
"""

# Import Libraries
import time
import numpy as np
from numpy.random import choice, shuffle
import matplotlib.pyplot as plt
from random import randint #  not taken from numpy so that it includes both endpoints
from collections import defaultdict, deque
from dataclasses import dataclass, field
try:
    from Modules import StateScriptInterface as ssi
    from Modules.ParameterFileInterface import ParameterFile
except:
    import StateScriptInterface as ssi
    from ParameterFileInterface import ParameterFile


"""
Enums
"""

class port_groups:
    """A Pseudo-Enum for Storing the Port Groups Names"""
    pump = 'Pump'
    well = 'Well'

class well_states:
    """An Enum Containing the Reward Well States"""
    inactive = 0
    active = 1
    reward_given = 2


"""
Reward Wells
"""

class Pump(ssi.Port):
    """An Object which Manages Delivering Reward via a Pump (auto-generated when creating Wells)"""
    
    updates_enabled_default = True
    
    def __init__(self, port:int, reward_function:int, rewarded_pump_var_name:str = None, updates_off:bool = None):
        """
        Parameters
        ----------
        port : int
            The Port Number of the Pump (as used by the ECU).
        reward_function : int
            The Function Number (in StateScript) to trigger to Deliver a Reward.
        rewarded_pump_var_name : str, optional
            The name of the variable in StateScript which stores the 
            port number for the active pump. 
            The default is wells.default_reward_pump_var_name.
        updates_off : bool, optional
            When True, DIO updates for this pump will be disabled. (this can clean up the StateScript, 
            but makes it a little harder to detect if there are hardware issues)
            The default is Pump.updates_enabled_default
            
        """
        # Initialize the Port
        updates_off = ssi._null_coalescing(updates_off, not Pump.updates_enabled_default)
        super(Pump, self).__init__(port, port_groups.pump, dio_updates = not updates_off) #  calls Port.__init__()
        
        # Initialize Pump-Specific Attributes
        self.reward_function = ssi.functions[reward_function] if reward_function in ssi.functions else reward_function
        self.rewarded_pump_var_name = ssi._null_coalescing(rewarded_pump_var_name, wells.default_reward_pump_var_name)
        
        # Turn Off Updates
        if updates_off:
            self.updates_off()
    
    def deliver_reward(self):
        """Deliver a Reward Using the Stored Parameters"""
        ssi.set_var(self.rewarded_pump_var_name, self.port)
        ssi.trigger(self.reward_function)
    
    def __repr__(self):
        return f"<Pump {self.index}>"


class _wells(list):
    """A List of All Well Objects, Coupled With a Dictionary of Well Objects by Group"""
    
    def __init__(self, well_list:list = (), append_only = True):
        """Create a List of Wells/Dictionary of Grouped Wells"""
        super(_wells, self).__init__(well_list)
        self.groups = defaultdict(list)
        self.append_only = append_only
        
        # convenience variable
        self.default_reward_pump_var_name = 'rewardPump'
    
    def __contains__(self, key) -> bool:
        """Check if a Group Name or Well is in the List"""
        return (type(key) in (str, int, tuple) and key in self.groups) or super(_wells, self).__contains__(key)
    
    def append(self, well):
        """Append a Well to the List of Wells (and to the grouped well dictionary)"""
        super(_wells, self).append(well)
        self.groups[well.group].append(well)
    
    def pop(self, index:int = -1):
        """Remove and Return the Well at the Specified Index (be careful, this could break the well indices)"""
        if not self.append_only:
            well = super(_wells, self).pop(index)
            self.groups[well.group].remove(well)
        elif ssi.config.verbose:
            ssi.disp("Well cannot be removed from well list because the well list is append only")
    
    def remove(self, well):
        """Remove a Well from the List (be careful, this could break the well indices)"""
        if not self.append_only:
            super(_wells, self).remove(well)
            self.groups[well.group].remove(well)
        elif ssi.config.verbose:
            ssi.disp("Well cannot be removed from well list because the well list is append only")
    
    def __getitem__(self, key):
        """Get a Well(s) by Index or Group Name 
        
        This checks if the key maps to a group first, so using integers as 
        group names can lead to collisions/unexpected behavior. I recommend 
        not using integers as group names, but if you really want to do that 
        then I recommend using _wells.get_group(self, group:str) or 
        _wells.get_well(self, index:int) to avoid unexpected behavior.
        """
        if key in self.groups:
            return self.groups[key]
        else:
            return super(_wells, self).__getitem__(key)
    
    def __setitem__(self, *args, **kwargs):
        """Setting Items in this List is Forbidden"""
        if ssi.config.verbose: ssi.disp('Setting Items in the Well List is Forbidden')
    
    def get_group(self, group:str):
        """Get the List of All Wells in the Specified Group (only necessary if group names are integers)"""
        return self.groups[group]
    
    def get_well(self, index:int):
        """Get a Well by its Index (only necessary if group names are integers)"""
        return super(_wells, self).__getitem__(index)
    
    def clear(self):
        """Remove All the Wells from the List"""
        super(_wells, self).clear()
        self.groups.clear()
    
    def _get_well_list(self, well_list) -> list:
        if well_list == None:
            return self
        elif type(well_list) == Well:
            return [well_list]
        elif type(well_list) in (str, int, tuple):
            return self.groups[well_list]
        else:
            return well_list
    
    def activate(self, well_list:list = None, set_leds:bool = False, update:bool = False):
        """Activate All Wells in the Given List/Group"""
        # Activate the Wells
        for well in self._get_well_list(well_list):
            well.activate()
        
        # Set the LEDs
        if set_leds:
            for well in self._get_well_list(well_list):
                well.led.on()
        
        # Update the Wells
        if update: self.update(well_list)
    
    def deactivate(self, well_list:list = None, set_leds:bool = True, update:bool = False):
        """Deactivate All Wells in the Given List/Group"""
        # Get the Wells List
        well_list = self._get_well_list(well_list)
        
        # Activate the Wells
        for well in well_list:
            well.deactivate()
        
        # Set the LEDs
        if set_leds:
            for well in well_list:
                well.led.off()
        
        # Update the Wells
        if update: self.update(well_list)
    
    def set_leds(self, well_list:list = None, state:int = 1):
        """Set All the LEDs in the Given List/Group"""
        for well in self._get_well_list(well_list):
            well.set_output(state)
    
    def update(self, well_list = None):
        """Set the LEDs to their next output states (if set)"""
        for well in self._get_well_list(well_list):
            well.update_output()
    
    def configure(self, well_ports:list, pump_ports:list, port_params:dict):
        """
        Create the Well Objects Specified by the <ports> dictionary

        Parameters
        ----------
        well_ports : list
            A list of all the ports to use for the wells. E.g.
                well_ports = [7, 1, 2, 3, 4, 5, 6]
        pump_ports : list
            A list of all the ports to use for the pumps. E.g.
                pump_ports = [15, 9, 10, 11, 12, 13, 14]
        port_params : dict
            A mapping of group names to well parameters and indices. E.g.
                port_params = {
                    'home': ((1, 50), [0]), # cued = 1, reward_function = 50uL
                    'arms': ((0, 150), [1, 2, 3, 4, 5, 6]) # cued = 0, reward_function = 150uL
                }
        
        Together, the example parameters create a 6-arm maze with 1 home 
        well (ports 7 and 15), and 6 arm wells (ports 1-6 for the wells, 
        and ports 9-14 for the pumps). Note that the indices in <port_params> 
        are used to access the port numbers in both <well_ports> and <pump_ports>.
        Example Call:
            
            ssi.configure(1, {50:10, 100:11, 150:12}) #  setting the reward functions must be done first if you're using a reward function mapping (this also sets the stats function to 1)
            wells.configure(
                [7, 1, 2, 3, 4, 5, 6],
                [15, 9, 10, 11, 12, 13, 14],
                {
                    'home': ((1, 50), 1), # replacing "[0]" with "1" means that it takes the next 1 indices (0)
                    'arms': ((0, 150), 6) # replacing "[1, 2, 3, 4, 5, 6]" with "6" means that it takes the next 6 indices (1-6)
                }
            )

        """
        used_indices = 0
        next_index = 0
        for group, (params, indices) in port_params.items():
            # Build the List of Indices (if necessary)
            if type(indices) == int:
                n = indices
                indices = []
                while len(indices) < n:
                    if not used_indices & (1 << next_index):
                        indices.append(next_index)
                    next_index += 1
            
            # Create the Well Objects
            for i in indices:
                used_indices |= 1 << i
                Well(well_ports[i], pump_ports[i], group, *params, active_well_list = self)
    
    def forbid_simultaneous_pokes(self, well_list:list = None) -> None:
        """Prints an Error Message if 2+ IR Beams are Tripped at the Same Time"""
        if well_list == None:
            ssi.config.forbid_simultaneous_inputs(port_groups.well)
        else:
            port_list = [well for well in self._get_well_list(well_list)]
            ssi.config.forbid_simultaneous_inputs(port_list)
    
    def forbid_simultaneous_rewards(self, well_list:list = None) -> None:
        """Prints an Error Message and Stops Reward Delivery if 2+ Pumps are Running at the Same Time"""
        if well_list == None:
            ssi.config.forbid_simultaneous_outputs(port_groups.pump)
        else:
            port_list = [well.pump for well in self._get_well_list(well_list)]
            ssi.config.forbid_simultaneous_outputs(port_list)

# Create an Alias for the Well List
WellList = _wells

# Create the Default Well List
wells = _wells()


class Well(ssi.Port):
    """An Object which Manages Reward Wells (controlling the LED, the IR Beam, and the Pump)
    
    Global Attributes
    -----------------
    reward_cooldown : float (seconds)
        Setting this to a value >0 will assert that a reward well cannot deliver a reward 
        if any other reward well delivered a reward <reward_cooldown seconds ago.
        The default is -1 (disabled)
    last_reward : float (seconds)
        The last time any well delivered a reward (set using time.time())
    
    Well Attributes
    ---------------
    group : str
        The name of the group the reward well belongs to (e.g. 'home', 'arms')
    led : Well
        led = self - is an alias of the current object to help make code more 
        readable. E.g. it lets you write "well.led.on()" rather than "well.on()"
    pump : Pump
        The Pump object which controls the pump that delivers reward to the well.
    cued : bool
        Whether or not the well is cued. Wells that are cued will update their 
        LED state on well.activate() and well.deactivate()
    state : int
        0 - inactive (the well will not give reward)
        1 - active (the well will give reward)
        2 - the well was active, and gave a reward (the well will not give a reward)
    reward_given : bool
        Whether or not the well has given a reward since the last time the well was activated.
        (only resets on well.activate())
    pokes : int
        The number of times the well has been poked (this epoch)
    rewards : int
        The number of rewards the well has delivered (this epoch)
    delay_outputs : bool
        When True, the output of the DIO port is only updated on well.update().
        (This is actually inherited from StateScriptInterface.Port) - The default is True.
    
    Port Attributes
    ---------------
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
    
    Notes
    -----
    Most reward well functionality is done more easilly with 'wells' (a default instance of class _wells).
    
    """
    
    # can be used as a safety check to make sure rewards aren't being given impossibly fast
    # which could be a sign of broken hardware
    last_reward = 0
    reward_cooldown = -1
    
    def __init__(self, well_port:int, pump_port:int, group:str, cued:bool, reward_function:int, rewarded_pump_var_name:str = None, active_well_list:_wells = wells):
        """I recommend initializing wells with wells.configure() rather than making each well on its own
        
        Parameters
        ----------
        well_port : int
            The Port Number of the IR Beam/LED (as used by the ECU).
        pump_port : int
            The Port Number of the Pump (as used by the ECU).
        cued : bool
            Whether or not the LED sate should be tied to whether or not the well is active.
        group : str
            The group of wells this well belongs to (it is recommended to use strings, but 
            <group> can be any hashable object).
        reward_function : int
            The Function Number (in StateScript) to trigger to Deliver a Reward.
        rewarded_pump_var_name : str, optional
            The name of the variable in StateScript which stores the 
            port number for the active pump. The default is Well.default_reward_pump_var_name.
        active_well_list : _wells, optional
            MODIFY THIS WITH CAUTION. Overriding this can allow you to have multiple 
            separate well lists (e.g. if you're controlling multiple environments 
            simultaneously from a single computer/ECU), but then it is on you to 
            manage those well lists.
        """
        
        # Set the Parameters
        super(Well, self).__init__(well_port, port_groups.well, delayed_outputs = True)
        self.group = group
        self.led = self # alias which can maybe make code more readable? (e.g. "well.led.on()" is more clear than "well.on()"?)
        self.pump = Pump(pump_port, reward_function, rewarded_pump_var_name)
        self.cued = cued
        self.state = well_states.inactive
        self.pokes = 0
        self.rewards = 0
        self.reward_given = False
        
        # Add the Well to the List of Wells
        active_well_list.append(self)
        
        # Log that the Well Was Initialized
        ssi.log(f'Well {self.index} Initialized: [group = {self.group}, led_port = {self.port}, pump_port = {self.pump.port}, cued = {self.cued}, reward_function = {self.pump.reward_function}]')
    
    def __repr__(self):
        return f"<Well {self.index}>"
    
    def __lt__(self, other): # enables sorting wells by index
        return self.index < other.index
    
    def reward(self, override_checks:bool = False):
        """Deliver a Reward (using the paramters given at well initialization)"""
        self.pokes += 1
        self.reward_given = False
        if (self.state == well_states.active and time.time() - Well.last_reward > Well.reward_cooldown) or override_checks:
            Well.last_reward = time.time()
            self.rewards += 1
            self.reward_given = True
            self.pump.deliver_reward()
            self.state = well_states.reward_given
            ssi.disp(f"well {self.index} poked; reward given = 1")
        elif self.state == well_states.active:
            ssi.disp(f"""reward not delivered to active well {self.index} [port {self.pump_port}] 
                 because not enough time has elapsed since the last reward. 
                 Please check that reward wells are functioning well""".replace('\n', ''))
        elif self.state == well_states.reward_given:
            ssi.disp(f"well {self.index} must be reactivated before it can give a second reward")
        else:
            ssi.disp(f"well {self.index} poked; reward given = 0")
        return self.reward_given
    
    def activate(self, update_leds_immediately:bool = False):
        """An Active Well will Deliver Reward when Poked"""
        self.state = well_states.active
        self.reward_given = False
        if self.cued or update_leds_immediately: self.led.on(update_leds_immediately)
    
    def deactivate(self, update_leds_immediately:bool = False):
        """An Inactive Well will not Deliver Reward when Poked"""
        self.state = well_states.inactive
        if self.cued or update_leds_immediately: self.led.off(update_leds_immediately)

# A Callback Function which is Good for Reward-Well Mazes (like the ones we currently make/use)
def well_callback(line:str) -> int:
    """Parse a Line from StateScript, returning if a command 
    was matched (res&1==1), and if reward was given (res&2==2)
    """
    
    # Use the Base Callback Function
    match, t, command, value = ssi.parse_command(line)
    
    # Check the Result
    if match:
        # Will be set to True if a reward is given
        reward_given = False
        
        # Check if a Well was Specified
        well = None
        if value != None:
            # Get the Well
            well = wells[ssi.config.get_port_index(value)]
            
            # Give Reward (well.reward() checks if the well is active)
            if command == ssi.config.rewarded_command:
                reward_given = well.reward()
        
        # Call the Command Handler
        if well != None:
            ssi.config.command_handlers[command](t, well)
        else:
            ssi.config.command_handlers[command](t)
        
        # Display Stats
        ssi.print_stats()
        
        # Return if a Match was Found
        return 1 + 2*reward_given
    else:
        # Return that a Match was Not Found
        return 0

"""
Platforms for Building Mazes
"""

class State(str):
    """Class for Defining StateMachine States"""

@dataclass(frozen = True)
class Transition:
    """Class for Defining StateMachine Transitions"""
    source_state: State
    target_state: State
    opcode: str
    condition: callable = field(default_factory = lambda : Transition.no_condition)
    action: callable = field(default_factory = lambda : Transition.null_action)
    
    def __hash__(self):
        return hash(self.target_state)
    
    def no_condition(*args, **kwargs):
        return True
    
    def null_action(*args, **kwargs):
        pass

class StateMachine:
    def __init__(self, start:State = 'start', end:State = 'end', timeout:int = 0):
        """
        Initialize a Finite State Machine

        Parameters
        ----------
        start : str or State, optional
            The starting state for the machine. 
            The default is 'start'.
        end : str or State, optional
            The ending state for the machine. 
            The default is 'end'.
        timeout : int, optional
            The number of minutes the machine 
            will run before timing out. 
            The default is 0 (no timeout).

        """
        
        self.states = {}
        self.transitions = {}
        self.start = start
        self.end = end
        self.state = self.start
        self.t0 = time.time()
        self.timeout = 60 * timeout
        self._timed_out = None
    
    def __setattr__(self, name:str, value:any):
        if name in ('start', 'end'):
            # make sure State objects are treated as such
            self.__dict__[name] = self._add_state(value)
        elif name == 'state':
            # Assert that the State is a Recognized State Object
            if value in self.transitions:
                # casting to make sure it's a State object and not a string
                self.__dict__[name] = self._add_state(value)
            else:
                raise ValueError(f'Unrecognized State: {value}')
        else:
            # otherwise just set the attribute
            self.__dict__[name] = value
    
    def __repr__(self):
        states = ', '.join(self.states.keys())
        return f"StateMachine[{states}]"
    
    def __getitem__(self, state:str):
        return self.transitions[str(state)]
    
    def __setitem__(self, state:str, transitions:list):
        self.transitions[str(state)] = transitions
    
    def __contains__(self, state:str):
        return str(state) in self.transitions
    
    def __bool__(self):
        return True
    
    def keys(self):
        return self.transitions.keys()
    
    def values(self):
        return self.transitions.values()
    
    def items(self):
        return self.transitions.items()
    
    def _add_state(self, state:str):
        state = state if isinstance(state, State) else State(state)
        if state not in self:
            self[state] = []
            self.states[state] = state
        return self.states[state]
    
    def add_transition(self, source_state:State, target_state:State, opcode:str, condition:callable = None, action:callable = None) -> None:
        source_state = self._add_state(source_state)
        target_state = self._add_state(target_state)
        args = [source_state, target_state, opcode]
        if condition != None: args.append(condition)
        if action != None: args.append(action)
        self[source_state].append(Transition(*args))
    
    def __iadd__(self, transition:Transition):
        self._add_state(transition.source_state)
        self._add_state(transition.target_state)
        self[transition.source_state].append(transition)
        return self
    
    def running(self) -> bool:
        """Returns if the State Machine is Still Running"""
        return self.state != self.end
    
    def timed_out(self) -> bool:
        """Returns if the Machine has or did Time Out"""
        return (time.time() - self.t0) > self.timeout if self._timed_out == None else self._timed_out
    
    def change_state(self, new_state:State) -> bool:
        """
        WARNING: Does NOT check if a valid transition exists
        
        This method mainly exists so that classes which inherit 
        from this class can override the change_state
        
        Note that this method checks for timeout, and disallows state 
        changes if the machine is in the end state
        
        Returns if the Machine is Running (for convenience)
        """
        if self.state == self.end:
            return False
        elif self.timeout and (time.time() - self.t0) > self.timeout:
            self.state = self.end
            self._timed_out = True
            return False
        else:
            self.state = new_state
            if new_state == self.end:
                self._timed_out = False
                return False
            else:
                return True
    
    def update(self, opcode:str, *args, **kwargs) -> bool:
        """Updates the State of the State Machine, Returning if the State Changed"""
        if self.running():
            for transition in self[self.state]:
                if opcode == transition.opcode and transition.condition(*args, **kwargs):
                    transition.action()
                    self.change_state(transition.target_state)
                    return True
        return False
    
    def stop(self):
        """Stop the State Machine"""
        self.state = self.end

"""
class RewardWellMaze(StateMachine):
    def __init__(self, start = 'start', timeout:int = 70, auto_ready:bool = True):
        super(RewardWellMaze, self).__init__(start, timeout = timeout)
        self.last_poke = None
        
        # Configure the Command Handlers
        ssi.config.rewarded_command = 'UP'
        ssi.config.add_commands({
            'READY': self.update,
            'UP' : self.update,
            'DOWN' : self.down,
            'LOCKEND' : self.update,
            'PING' : self.check_timeout
        })
        
        # Call the Ready Function
        if auto_ready: self.update('READY')
    
    def ready(self, t:int):
        pass
    
    def down(self, t:int, well:Well):
        wells.update()
    
    def close(self, t:int):
        if self.running():
            self.change_state(self.end)
    
    def check_timeout(self, t:int, *args, **kwargs):
        if self.timed_out(): self.close()
    
    def check_success_rate(self):
        pass
    
    def callback(self, line:str):
        if self.running() and well_callback(line):
            self.check_success_rate()
        elif not self.running() and ssi.command_is_valid(line):
            ssi.print_stats()
            self.check_success_rate()
        
#"""

class FileDrivenMaze(ParameterFile):
    def __init__(self, filepath:str):
        # Initialize Attributes which Aren't Calculated From the Parameter File
        self.goal = None
        self.previous_goal = None
        self.reps_remaining = 0
        self.start_time = None
        self.timeout_grace_period = 0
        self.max_epochs_updated = False
        self.leds = []
        
        # Load the Parameter File (initializes several attributes)
        self.first_goal = True
        self.last_goal = None
        self.alpha = 0.5
        self.beta = -0.5
        self.gamma = 0.8
        self.delta = 0.01
        super(FileDrivenMaze, self).__init__(filepath)
        
        # Pre-Compute Adjusted Versions of the Arm Selection Parameters
        arms = self.arms if hasattr(self, 'arms') else len(self.rewarded_visits)
        self.delta_prime = self.delta / (1 - arms * self.delta)
        outreps = sum(self.outreps)/len(self.outreps) if hasattr(self.outreps, '__iter__') else self.outreps
        trials = self.max_trials if self.max_trials > -1 else outreps * self.goal_blocks / self.success_threshold
        self.gamma_prime = self.gamma ** (1 / trials)
        
        # Check for Bad Parameter Combos
        if self.cues and max(self.goals, self.forageassist) > abs(self.cues):
            goal_str = "goals" if self.goals >= self.forageassits else "forageassisted goals"
            goals = max(self.goals, self.forageassist)
            raise ValueError(f"Number of Cues ({abs(self.cues)}) must be greater than the number of {goal_str} ({goals})")
        
        # Initialize Attributes which Are Calculated from the Parameter File
        self.possible_goal_count = len(self.total_visits)
    
    def get_epoch_time(self) -> float:
        """calculates the time since the first home poke in minutes"""
        return (time.time() - self.start_time) / 60 if self.start_time != None else 0
    
    def timed_out(self) -> float:
        """checks if the epoch timed out (returns true if enough time has passed regardless of the number of trials completed)"""
        return self.get_epoch_time() > self.timeout + self.timeout_grace_period
    
    def get_outreps(self):
        if type(self.outreps) == int:
            return self.outreps
        elif type(self.outreps) == tuple and len(self.outreps) == 2:
            return randint(*self.outreps)
        elif type(self.outreps) == list and self.outreps:
            return choice(self.outreps)
        else:
            outreps = 1 if self.cued == 1 else 15
            ssi.disp(f"unrecognized value for outreps. defaulting to {outreps}")
            return outreps
    
    def updated_weighted_visits(self, well_index:int, rewarded:bool):
        w = self.alpha if rewarded else 1 - self.alpha
        for i in range(len(self.weighted_visits)):
            self.weighted_visits[i] *= self.gamma_prime
        self.weighted_visits[well_index - 1] += w * (1 - self.gamma)
    
    def get_goal_probabilities(self, possible_goals:list):
        """Get the Selection Probability for Each Possible Goal"""
        # Check the Goal Selection Mode
        if self.goal_selection_mode <= 0:  # Choose a Goal(s) at Random (but not the previous goal)
            P = np.ones(len(possible_goals), int)
        elif self.goal_selection_mode == 1:
            rewarded_visits = np.array(self.rewarded_visits)
            total_visits = np.array(self.total_visits)
            visits = self.alpha * rewarded_visits + (1 - self.alpha)*(total_visits - rewarded_visits)
            return softmax(visits - np.min(visits) + 1, self.beta)
        elif self.goal_selection_mode == 2:
            return softmax(np.array(self.weighted_visits) - np.min(self.weighted_visits) + 1, self.beta)
        else:
            raise ValueError(f"unrecognized arm selection mode: {self.goal_selection_mode}")
        
        # Apply the Lower Bound
        P += self.delta_prime
        
        # Exclude the Current Goal
        if self.goals == 1 and self.goal:
            P[self.goal[0].index] = 0
        
        # Normalize and Return the Probabilities
        return P / np.sum(P)
    
    def _disp_goal_block_info(self):
        # this ensures that this string is always idential (for ease of parsing later)
        ssi.disp(f"new goal block: [goal = {[w.index for w in self.goal]}, outreps = {self.reps_remaining}]")
    
    def select_goal(self, possible_goals:list):
        # Set the Previous Goal
        self.previous_goal = [] if self.goal == None else self.goal
        
        # Select the Number of Outreps
        self.reps_remaining = self.get_outreps()
        ssi.logger.add_line_break(char = '-')
        ssi.disp('new goal block', False)
        
        # Get the Possible Goals (if specified with a group)
        if type(possible_goals) in (str, int, tuple) and possible_goals in wells:
            possible_goals = wells[possible_goals]
        
        # Copy the Possible Goals (to not affect the original list)
        possible_goals = possible_goals.copy()
        
        # Get the Number of Goals to Select
        goals = max(self.goals, self.forageassist)
        
        # Check if All Options Need to be Selected
        if len(possible_goals) <= goals:
            self.goal = possible_goals
            self._disp_goal_block_info()
            return
        
        # Choose a Goal From the Goal Sequence
        def selection_mode_0():
            # Get the Next Goal
            goal = [wells[self.goal_sequence.pop()]]
            
            # Generate a New Sequence if Necessary
            if not self.goal_sequence:
                # Initialize a New Goal-Sequence
                self.goal_sequence = [goal.index for goal in possible_goals]
                shuffle(self.goal_sequence)
                
                # Re-Shuffle the Sequence to Make Sure the Next Goal isn't the Current Goal
                if len(self.goal_sequence) > 1:
                    while wells[self.goal_sequence[-1]] == goal[0]:
                        shuffle(self.goal_sequence)
            
            # Return the Selected Goal
            return goal
        
        # Check the Arm Selection Mode
        goal = None
        update_last_goal = self.goals == 1 and self.end_mode == 1 and self.last_goal != None
        if update_last_goal and self.first_goal:
            update_last_goal = False
            self.first_goal = False
            if self.last_goal and self.last_goal[-1]:
                goal = [possible_goals[self.last_goal[-1] - 1]]
            elif self.goal_selection_mode == 0:
                goal = selection_mode_0()
            else:
                goal = [choice(possible_goals)]
            self.last_goal.append(0)
        elif self.goal_selection_mode or self.goals != 1:
            # Get the Probability Distribution
            P = self.get_goal_probabilities(possible_goals)
            
            # Draw From the Distribution
            if self.goals == 1:
                # Draw Until Success
                goal = self.previous_goal
                while goal == self.previous_goal and len(possible_goals) > 1:
                    # Draw From the Distribution
                    goal = list(choice(possible_goals, 1, True, P))
                    
                    # Make the Distribution a Little Flatter
                    # (helps avoid cases where the distirbution is so skewed so badly that only 1 goal can be picked by np.choice)
                    P = 0.99 * P + 0.01
                    P /= np.sum(P)
            else:
                # Draw From the Distribution (it's ok for this to be the same as the previous goal)
                goal = list(choice(possible_goals, self.goals, False, P))
        else: # pre-scripted shuffled sequence (not really intended to be used for more than 1 goal)
            goal = selection_mode_0()
        
        # Make Sure the Goal is a List
        if type(goal) != list:
            if hasattr(goal, '__iter__'):
                goal = list(goal)
            else:
                goal = [goal]
        
        # Select Extra LEDs to be Lit
        if self.cues and abs(self.cues) < len(possible_goals):
            self.leds = goal.copy()
            if abs(self.cues) > len(self.leds):
                leds = [w for w in wells if w not in goal and w != self.home]
                if self.previous_goal:
                    prev = self.previous_goal[0]
                    if prev in leds: 
                        leds.remove(prev)
                    if self.cues > 0:
                        self.leds.append(prev)
                needed = abs(self.cues) - len(self.leds)
                if needed > 0:
                    if needed < len(leds):
                        self.leds.extend(choice(leds, needed, False))
                    else:
                        self.leds.extend(leds)
        elif self.cues:
            self.leds = possible_goals.copy()
        else:
            self.leds = []
        
        # Track the Selected Goal
        self.goal = goal
        if update_last_goal: 
            self.last_goal[-1] = self.goal[0].index
        self.stats.this_goal = 0
        for g in goal: self.goal_counts[g.index - 1] += 1
        self._disp_goal_block_info()
        
        # Return the Selection
        return goal
    

"""
Live Plot Data
"""

class Poke:
    def __init__(self, well:int, rewarded:bool, t_up:int, t_down:int = None):
        self.well = well
        self.rewarded = rewarded
        self.up = t_up
        self.down = t_down
    
    def __repr__(self) -> str:
        return f"Poke[{self.well} {int(self.rewarded)}]"

class Trial:
    def __init__(self, number:int, state:tuple = None, events:list = ()):
        self.number = number
        self.state = state
        self.events = list(events)
        self.start = None
        self.end = None
        self.set_times()
    
    def __repr__(self) -> str:
        return f"Trial({self.number}){self.events}"
    
    def set_times(self) -> None:
        """Compute/Record the Start/End Times of the Trial (typically not necessary to call by hand)"""
        if self.events:
            self.start = self.events[0].up
            self.end = self.events[-1].down
            if self.end == None:
                self.end = self.events[-1].up
    
    def up(self, well:int, rewarded:bool, t:int) -> Poke:
        """Add a New Poke to the Trial"""
        self.events.append(Poke(well, rewarded, t))
        self.end = t
        return self.events[-1]
    
    def down(self, t:int) -> Poke:
        """Update the Down Time for the Most Recent Poke"""
        if self.events:
            self.events[-1].down = t
            self.end = t
            return self.events[-1]

class Block:
    def __init__(self, state:tuple = None, start:int = None, end:int = None, trials:list = ()):
        self.state = state
        self.trials = list(trials)
    
    def new_trial(self, state:tuple = None, pokes:list = ()) -> Trial:
        """Add a New Trial to the Block"""
        trial = Trial(len(self.trials) + 1, state, pokes)
        self.trials.append(trial)
        return trial
    
    def up(self, well:int, rewarded:bool, t:int) -> Poke:
        """Create a New Poke"""
        self.end = t
        if self.trials:
            return self.trials[-1].up(well, rewarded, t)
    
    def down(self, t:int) -> Poke:
        """Update the Down Time for the Most Recent Poke"""
        self.end = t
        if self.trials: 
            return self.trials[-1].down(t)

class Plotter:
    def __init__(self, expected_trials:int, title:str, filepath:str = None, to_table_data = None, first_line:str = None, state:tuple = (), plot:bool = True):
        self.blocks = []
        self.trials = [Trial(0, state)]
        self.first_trial = True
        
        self.complete = False
        self.plot = bool(plot)
        self.active = True
        
        self.filepath = filepath
        self.to_table_data = to_table_data
        self.logging = filepath != None and hasattr(to_table_data, '__call__')
        self._open_log(first_line)
        self.trials_logged = 0
        
        self.unplotted_pokes = deque()
        
        self.title = title
        self.x = 1
        self.X = expected_trials
        self.fig = None
        self.ax = None
        self.init_plot()
    
    def _open_log(self, first_line:str) -> None:
        if self.logging:
            with open(self.filepath, 'w') as f:
                f.write(first_line)
                f.write('\n')
                f.close()
    
    def _log(self, trial) -> None:
        if self.logging and self.active:
            with open(self.filepath, 'a') as f:
                f.write(self.to_table_data(trial))
                f.write('\n')
                f.close()
            self.trials_logged += 1
    
    def _close_log(self):
        if self.logging and self.active:
            while self.trials_logged < len(self.trials):
                self._log(self.trials[self.trials_logged])
            self.logging = False
    
    def _try_plot(self, method, *args):
        """a wrapper to make sure plotting never breaks anything"""
        try:
            if self.plot and self.active:
                method(*args)
        except:
            print('warning: unable to initialize the live plot; live plot disabled')
            self.plot = False
    
    def _init_plot(self):
        
        def on_close(event):
            self.plot = False
        
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('close_event', on_close)
        plt.suptitle(self.title + ": Running")
        plt.ylabel('Well Number')
        plt.xlabel('Trials + Lockouts')
        self._set_axis_params()
        plt.show()
    
    def _set_axis_params(self):
        plt.ylim(0.8, 6.2)
        self.fig.set_figheight(5)
        self._set_xaxis_lims()
        plt.ylabel('Arm Number')
        plt.xlabel('Trials + Lockouts')
    
    def _set_xaxis_lims(self):
        try:
            if self.plot and self.active:
                plt.xlim(0, self.X)
                self.fig.set_figwidth(self.X / 6)
        except:
            pass
    
    def _update_plot_full(self):
        self.ax.cla()
        self._set_axis_params()
        self.x = self.raster_plot(axes = self.ax, included = 1, black_line = False)
        if self.x > self.X:
            self.X = self.x
            self._set_axis_params()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _update_plot(self):
        while self.unplotted_pokes:
            well, rewarded, lockout = self.unplotted_pokes.popleft()
            marker = '.' if rewarded else '+'
            color = 'grey' if lockout else 'magenta'
            plt.errorbar(self.x, well, fmt = marker, color = color)
            self.x += 1
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()
    
    def _add_block_divider(self):
        self.update_plot() # this is necessary to make sure self.x is accurate
        if self.maze.outreps != 1 and self.x > 1:
            self.ax.plot([self.x - 0.5]*2, [0, 8], color = 'lightgrey')
    
    def init_plot(self):
        self._try_plot(self._init_plot)
    
    def add_block_divider(self):
        self._try_plot(self._add_block_divider)
    
    def update_plot(self):
        self._try_plot(self._update_plot)
    
    def new_block(self, t:int, state:tuple = ()) -> None:
        # Clean Up the Current Block
        if self.blocks:
            self.add_block_divider()
        
        # Add a New Block
        self.blocks.append(Block(state, t))
    
    def new_trial(self, t:int, state:tuple = ()) -> None:
        # delete the 0th trial
        if self.first_trial and not self.trials[0].events:
            self.trials.pop(0)
        self.first_trial = False
        
        # log the last trial
        if self.trials:
            self._log(self.trials[-1])
        
        # make sure there's a block to add trials to
        if not self.blocks: self.new_block(state)
        
        # create/add the trial
        self.trials.append(self.blocks[-1].new_trial(state))
    
    def add_event(self, event:any) -> None:
        """Add an arbitrary event to the current trial (plots nothing)"""
        self.trials[-1].events.append(event)
    
    def up(self, well:int, rewarded:bool, t:int, show:bool = True, lockout:bool = False, state:tuple = ()) -> None:
        # record the poke
        if self.blocks:
            self.blocks[-1].up(well, rewarded, t)
        else:
            self.trials[-1].up(well, rewarded, t)
        
        # plot the poke
        if show:
            # Check for a Lockout
            if lockout:
                # Update the Plot Limit
                self.X += 1
                self._set_xaxis_lims()
            
            # Add the Poke ot the Plot
            self.unplotted_pokes.append((well, rewarded, lockout))
            self.update_plot()
    
    def down(self, t):
        if self.blocks: self.blocks[-1].down(t)
        
    def close(self):
        self._close_log()
        self.update_plot()
        if self.plot and self.active:
            plt.suptitle(self.title + ": Complete")
        self.active = False

"""
Other Functions
"""

# softmax distribution function with a "temperature" parameter
def softmax(X, beta):
    """returns the normalized vector e^(beta * X)"""
    X = np.exp(beta * np.array(X))
    return X / np.sum(X)