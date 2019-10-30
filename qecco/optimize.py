from .system.system import System
from .autocodec.autocodec import Autocodec

# # FIXME:  Get function to work with system-class
def optimize(opt_type=None, parameters=None):
    print("Oh noes!")
#     accepted_types = ["encoder", "decoder", "autocodec", None]

#     if opt_type in accepted_types:
#         if opt_type is None:
#             opt_type_inputs()
#         elif opt_type == "encoder":
#             try:
#                 if parameters["simple_cost"] is True:
#                     Encoder(parameters).build_codes().optimize()
#                 else:
#                     Encoder(parameters).build_codes().build_rhos().optimize()
#             except TypeError:
#                 Encoder(parameters).build_codes().build_rhos().optimize()
#         elif opt_type == "decoder":
#             Decoder(parameters).build_rhos().apply_loss().optimize()
#         elif opt_type == "autocodec":
#             encoder = Encoder()
#             decoder = Decoder()
#             Autocodec(encoder, decoder).optimize()
#     else:
#         print("Optimization method not found")


# def opt_type_inputs():
#     opt_type = input("Enter type of encoder/decoder: ")

#     if opt_type:
#         print("Write nice input questions!")
#     optimize(opt_type)
