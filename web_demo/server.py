

def _create_scene():
    pass


@app.route('/')
def index():
    # UI: upload RGB D
    # UI: starting_pos (default value) + end_pos(default value)
    # UI: text prompt (for grasp3d service)
    # UI: max_planning_obj: 3

    # server side: call grasp3d service get the data, planing for max score of max_planning_obj
    # return and visualize(rgb): 
    return render_template('index.html')



@app.route('/predict/')
def predict():
    # input:
    #   rgb, optional
    #   dpt
    #   objs: each obj has attribute: [mask, affordance, bbox, score]
    #   start_pos: pose of end-effector, (x-m, y-m, z-m, roll-rad, pitch-rad, yaw-rad)
    #   end_pos: pose of end-effector aftering picking:  (x-m, y-m, z-m, roll-rad, pitch-rad, yaw-rad)
    #   excution_simulation: bool
    #
    # find the best picking candidate in the objs, and return the best one (index)
    # steps:
        # - destory scene
        # - create scene
        # - plan
        # - destory scene
    # return:
    # - obj_index: int
    # - pick_pos: (x-m, y-m, z-m, roll-rad, pitch-rad, yaw-rad)
    # - trajectory: dict
        # - approaching: list of joints angles, [[0,]* 7, ..., ]
        # - grasp-approach: list of joints angles
        # - retreat: 
        # - carrying:
        # - returning:




def get_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path', default='configs/dinox_core/demo.py')
    parser.add_argument('--model-name', help='the name of model', default='full')
    parser.add_argument('--load-from', help='checkpoint file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--host', type=str, help='the host addr to run the server', default='0.0.0.0')
    parser.add_argument('--port', help='the port to run the server', default=12086)
    parser.add_argument('--debug', help='start debug model', action='store_true', default=False)
    parser.add_argument('--test', help='test the app', action='store_true', default=False)
     
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    if args.test:
        # local test
        pass
    else:
        # Run server - only watch server.py for changes
        app.run(host=args.host, port=args.port, debug=args.debug, extra_files=[__file__])