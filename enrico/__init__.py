def get_model(args): 
    if args.model_type == "jlogits": 
        from enrico.joint_model import MultimodalEnricoModel
    elif args.model_type == "ensemble": 
        from enrico.ensemble_model import MultimodalEnricoModel
    else: 
        raise NotImplementedError("Model type not implemented")
    
    return MultimodalEnricoModel(args)
