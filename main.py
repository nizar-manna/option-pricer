"""Option Pricer

This app calculates the price of a European vanilla call/put option. 
Defines "Option" class, taking Black-Scholes parameters. 
The price_BS() method computes the option's price.

User interface allows parameter input, collected in a dictionary, 
then passed to an Option instance. Option price is obtained by calling 
price_BS().

Libraries/Modules used:
NumPy: for log, sqrt, exp, divide methods
from scipy.stats import norm: Cumulative distribution function of 
the standard normal distribution
    """

import numpy as np
from scipy.stats import norm

VERSION = 1.0

# Global variables used to compute time_unit.
# Simply the number of days/weeks/... in a year,
# so that time_to_expiration can be expressed in years
DAYS_IN_YEAR = 365
WEEKS_IN_YEAR = 52+(1/52)
VALID_TIME_UNITS = {"d": DAYS_IN_YEAR, "w": WEEKS_IN_YEAR, "y": 1}


class Option:
    """Class representing an european vanilla option.

    Contains all parameters for Black-Scholes option pricing formula 
    Includes pricer_BS() method, calculating the option's price with those
    parameters.
    """
    def __init__(self, is_call=None, underlying_price=None,
                 time_unit=None, time_to_expiration=None, 
                 strike=None, rfree_discrete=None, 
                 volatility=None, option_parameters=None):

        """Class constructor

        Args:
            is_call (bool, optional): Call option if True, Put option if False. 
                Defaults to None.
            underlying_price (float, optional): Underlying asset price. 
                Defaults to None.
            time_unit (int, optional): Time unit chosen by the user, expressed
                as units per year.
                e.g. if the user choses "days", the time_to_expiration
                will then be expressed in days.
                Defaults to None.
            time_to_expiration (float, optional): Time until 
                the option expires, inputted in days/months/years,
                then divided by time_unit to be expressed in years.
                Defaults to None.
            strike (float, optional): Strike price. Defaults to None.
            rfree_discrete (float, optional): Rate of return on
                a riskless asset (aka risk free rate)
                Defaults to None.
            volatility (float, optional): Implied volatility of the
                underlying asset.
                Defaults to None.
            option_parameters (dict, optional): Dictionary containing 
                all the above arguments, adding another way
                to initialize an instance.
                Defaults to None.

        """
        if option_parameters is not None:
            self.is_call = option_parameters.get('is_call')[1]
            self.underlying_price = \
                option_parameters.get('underlying_price')[1]
            self.strike = option_parameters.get('strike')[1]
            self.time_unit = option_parameters.get('time_unit')[1]
            self.time_to_expiration = \
                option_parameters.get('time_to_expiration')[1] / self.time_unit
            self.rfree_discrete = option_parameters.get('rfree_discrete')[1]
            self.volatility = option_parameters.get('volatility')[1]

        else:
            self.is_call = is_call
            self.underlying_price = underlying_price
            self.time_unit = time_unit
            self.time_to_expiration = time_to_expiration / time_unit
            self.strike = strike
            self.rfree_discrete = rfree_discrete
            self.volatility = volatility

        # Convert discrete risk-free rate to continuous
        # using logarithmic transformation
        self.rfree_continuous = np.log(1 + self.rfree_discrete)

    def pricer_BS(self):
        """Pricing the option using the Black and Scholes formula.

        Returns:
            float: Price of the option.
        """

        # Computing D1: (ln(S / K) + t) * (r + (Sigma^2 / 2)) / (Sigma * Sqrt(t))
        # Where S = underlying_price, K = strike, t = time_to_expiration
        # r = rfree_continuous, Sigma = volatility
        d1 = ((np.log(self.underlying_price / self.strike) 
               + (self.time_to_expiration) 
              * (self.rfree_continuous 
                   + (np.divide(self.volatility ** 2, 2))))
              / (self.volatility * np.sqrt(self.time_to_expiration)))

        # We obtain d2 by substracting (Sigma - Sqrt(t)) from d1
        d2 = d1 - (self.volatility * np.sqrt(self.time_to_expiration))

        # Checks whether the option is a call (True) or a put (False)
        # and returns a price accordingly, using Black-Scholes formula.
        # Returns a message if neither True or False are selected.
        # norm.cdf = Cumulative distribution function of the standard normal
        # distribution
        if self.is_call is True:
            call_price = (self.underlying_price * norm.cdf(d1) 
                          - self.strike 
                          * np.exp(- self.rfree_continuous
                                   * self.time_to_expiration)
                          * norm.cdf(d2))
            return call_price

        if self.is_call is False:
            put_price = (self.strike 
                         * np.exp(- self.rfree_continuous
                                  * self.time_to_expiration)
                         * norm.cdf(-d2) 
                         - self.underlying_price * norm.cdf(d1))
            return put_price
            
        else:
            print("Error: is_call should be either True or False")
            return None


def boot_menu():
    """A simple menu shown at each boot, showcasing app name,
    author name, github, type of option priced.
    """
    print(f"Financial options pricer v{VERSION}\n"
          "Author: Nizar MANNA\n"
          "github: nizar-manna")
    print("European vanilla options pricing")


def is_call_input():
    """Asks the user whether the option should be a (C)all or a (P)ut

    Returns:
        bool: True if the choice is call, False if the choice is put.
    """
    is_call = input("C for a Call option, P for a Put: ").lower()

    while True:

        if is_call not in ["c", "p"]:
            print("Please input either C (Call) or P (Put)")
        
        elif is_call == "c":
            return True
        
        elif is_call == "p":
            return False


def time_unit_input():
    """Asks the user to chose between different time units

    Returns:
        string: a string of a time_unit, used to access a value in VALID_TIME_UNITS. 
    """
    time_unit = ""

    while time_unit not in list(VALID_TIME_UNITS.keys()):
        print("Please input a time unit")
        time_unit = input("days (D),"
                          "weeks (W), years (Y): ").lower()

        if time_unit not in list(VALID_TIME_UNITS.keys()):
            print("error: incorrect input.")

        else:
            return time_unit


def parameters_input():
    """Lets the user input each parameter needed to create an Option object
    able to compute its price using Black Scholes formula:
    underlying_price, strike, time_unit, volatility, rfreediscrete, is_call


    Returns:
        dict: A dictionary comprising all the above-mentioned parameters.
              Should be passed as an argument when creating an Option instance.
    """


    # We create an empty dictionary following this model:
    # {"parameter name": ["extended name", value]}
    # The dictionary key is used to reach and cycle conveniently
    # through parameters. We use a list comprised
    # of the extended name (for display purposes) and the value.

    option_parameters = {"underlying_price": ["underlying asset price", None],
                     "strike": ["strike (price of exercise)", None],
                     "time_unit": ["time frame", None],
                     "time_to_expiration": ["time to expiration", None],
                     "volatility": ["volatility", None],
                     "rfree_discrete": ["risk-free rate (discrete)", None],
                     "is_call": ["whether the option is a call (C) \
                                 or a put (P) ", None]}

    # The user inputs a value for each parameter needed to compute the B-S
    # formula. Those parameters are then stored
    # in the option_parameters dictionary
    for variable in option_parameters:

        if variable == "is_call":
            option_parameters[variable][1] = is_call_input()

        elif variable == "time_unit":
            option_parameters[variable][1] = \
                VALID_TIME_UNITS[time_unit_input()]

        else:
            try:
                option_parameters[variable][1] = float(input(
                f"Please input {option_parameters[variable][0]}: "))

            except ValueError:
                print("error: wrong input (should be a number)")

    return option_parameters


def main_loop():
    """Main loop which runs the program.
    """

    restart = ""

    # Shows title, version and credentials
    boot_menu()

    # Creates an Option, asks the user for its parameters 
    # and then shows its price.
    # Keeps looping until the user specifies he doesn't want to restart ('n')
    while restart != "n":
        NewOption = Option(option_parameters=parameters_input())
        print("-------------------------")
        print(f"Option Price: {NewOption.pricer_BS()}")
        print("\n \n")


        # The user is then asked whether he'd likes to continue.
        # The loop breaks when the user has inputted 'y' or 'n'.
        # The main loop breaks when the user has inputted 'n'.
        while restart != "n":
            restart = input("Continue with another option? "
                            "Y for Yes, N for No").lower()

            if restart in ["y", "n"]:
                break

            else:
                print("error: wrong input. Please input either Y or N")


main_loop()
