from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """ This class includes test options.
    
    Also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        
        # TEST PARAMETERS
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        
        # OUTCOME DIRECTORY PARAMETERS
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser
