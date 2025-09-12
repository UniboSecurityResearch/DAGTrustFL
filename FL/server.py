from loguru import logger
import numpy as np
import psutil
import gc
import os
import time
import torch
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client


# Import DAG modules from parent directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DAG'))
from DAG_functions import init_DAG
from DAG_functions import get_trust_vector
from DAG_functions import milestone
from parameter_utils import cosine_similarity_defense_score
from parameter_utils import cos_sim_simple
from parameter_utils import flatten_parameters
r_filez = []
w_filez = []

# Global timing tracking variables
ml_training_times = []
dag_operation_times = []
round_start_times = []
total_experiment_start_time = None


def check_memory():
    """
    Monitor memory usage and trigger garbage collection if needed.
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        available_mb = system_memory.available / 1024 / 1024
        
        #logger.info(f"Memory usage: {memory_mb:.2f} MB ({memory_percent:.1f}% of system)")
        #logger.info(f"Available system memory: {available_mb:.2f} MB")
        
        # Check GPU memory if CUDA is available
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                #logger.info(f"GPU memory - Allocated: {gpu_memory_allocated:.2f} MB, Reserved: {gpu_memory_reserved:.2f} MB")
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        # Trigger garbage collection if memory usage is high
        if memory_mb > 4000:  # Alert if over 4GB
            #logger.warning(f"High memory usage detected: {memory_mb:.2f} MB")
            #logger.info("Triggering garbage collection...")
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            collected = gc.collect()
            #logger.info(f"Garbage collection completed, collected {collected} objects")
            
            # Check memory again after cleanup
            process = psutil.Process(os.getpid())
            new_memory_mb = process.memory_info().rss / 1024 / 1024
            #logger.info(f"Memory after cleanup: {new_memory_mb:.2f} MB (freed {memory_mb - new_memory_mb:.2f} MB)")
        
        # Critical memory warning
        if memory_mb > 6000:  # Critical if over 6GB
            logger.warning(f"CRITICAL: Very high memory usage: {memory_mb:.2f} MB")
            logger.warning("Consider reducing batch size, model size, or number of workers")
        
        return {
            'memory_mb': memory_mb,
            'memory_percent': memory_percent,
            'available_mb': available_mb,
            'gpu_allocated_mb': gpu_memory_allocated if torch.cuda.is_available() else 0,
            'gpu_reserved_mb': gpu_memory_reserved if torch.cuda.is_available() else 0
        }
        
    except Exception as e:
        logger.error(f"Error checking memory: {e}")
        return None

def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    global ml_training_times, dag_operation_times, round_start_times
    
    # Start timing for this round
    round_start = time.time()
    round_start_times.append(round_start)
    
    # Memory check at start of round
    #logger.info(f"=== ROUND {epoch} START - Memory Check ===")
    #check_memory()
    
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)
    logger.info("Selected workers for epoch #{}: {}", str(epoch), str(random_workers))
    global_model = clients[0].get_nn_parameters()

    # === ML TRAINING PHASE - START TIMING ===
    ml_training_start = time.time()
    
    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)
    
    # === ML TRAINING PHASE - END TIMING ===
    ml_training_end = time.time()
    ml_training_time = ml_training_end - ml_training_start
    ml_training_times.append(ml_training_time)
    
    logger.info(f"🤖 ML Training time for round {epoch}: {ml_training_time:.3f} seconds")
    
    # Memory check after training
    #logger.info(f"=== ROUND {epoch} AFTER TRAINING - Memory Check ===")
    #check_memory()
    
    #Saving parameters of clients in a dict for the graph
    parameters_clients = {}
    for client_idx in random_workers:
        parameters_clients[client_idx] = clients[client_idx].get_nn_parameters()
    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    
    # === DAG OPERATIONS PHASE - START TIMING ===
    dag_operations_start = time.time()
    
    trust_vector, update_contributions = get_trust_vector(parameters_clients, epoch, logger=logger)
    
    # === DAG OPERATIONS PHASE - END TIMING ===
    dag_operations_end = time.time()
    dag_operations_time = dag_operations_end - dag_operations_start
    dag_operation_times.append(dag_operations_time)
    
    logger.info(f"🔗 DAG Operations time for round {epoch}: {dag_operations_time:.3f} seconds")
    
    # Memory check after DAG operations
    #logger.info(f"=== ROUND {epoch} AFTER DAG OPERATIONS - Memory Check ===")
    #check_memory()

    # Extract trust values only for the selected workers in this round
    ordered_parameters = []
    ordered_trust_values = []
    for client_idx in random_workers:
        if(update_contributions[client_idx] != 0):
            ordered_parameters.append(clients[client_idx].get_nn_parameters())
            ordered_trust_values.append(trust_vector[client_idx])
            #ordered_trust_values.append(1.0) Experiment without the trust component
        else:
            args.get_logger().info(f"Client {client_idx} has no updates contribution, skipping.")

    args.get_logger().info("Selected workers: {}", str(random_workers))
    args.get_logger().info("Corresponding trust values: {}", str(ordered_trust_values))
    
    new_nn_params = average_nn_parameters(ordered_parameters, ordered_trust_values)
    
    try:
        #GIACOMO: creating a milestone with the new global model parameters
        milestone(new_nn_params, epoch, logger=logger)
    except Exception as e:
        logger.error(f"Error in milestone creation for epoch {epoch}: {e}")
        # Memory check after error
        check_memory()
        raise e
    
    # Memory check after milestone
    #logger.info(f"=== ROUND {epoch} AFTER MILESTONE - Memory Check ===")
    #check_memory()
    
    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        try:
            client.update_nn_parameters(new_nn_params)
        except Exception as e:
            logger.error(f"Error in updating nn params for epoch {epoch} and client {client}: {e}")
            # Memory check after error
            check_memory()
            #raise e

    # Final memory check at end of round
    logger.info(f"=== ROUND {epoch} END - Memory Check ===")
    memory_stats = check_memory()
    
    # Log memory trend
    if hasattr(train_subset_of_clients, 'previous_memory'):
        if memory_stats:
            memory_increase = memory_stats['memory_mb'] - train_subset_of_clients.previous_memory
            logger.info(f"Memory change since last round: {memory_increase:+.2f} MB")
            
            # Warning if memory is consistently increasing
            if memory_increase > 50:  # More than 50MB increase
                logger.warning(f"Significant memory increase detected: {memory_increase:.2f} MB")
    
    if memory_stats:
        train_subset_of_clients.previous_memory = memory_stats['memory_mb']

    return clients[0].test(), random_workers

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    global total_experiment_start_time, ml_training_times, dag_operation_times, round_start_times
    
    # Initialize timing tracking
    total_experiment_start_time = time.time()
    ml_training_times = []
    dag_operation_times = []
    round_start_times = []
    
    logger.info("🕐 Starting timing tracking for DAGTrustFL experiment")
    
    # Delete existing result files if they exist
    if r_filez and os.path.exists(r_filez[0]):
        os.remove(r_filez[0])
    if w_filez and os.path.exists(w_filez[0]):
        os.remove(w_filez[0])
    
    epoch_test_set_results = []
    worker_selection = []
    
    # Initial memory baseline
    logger.info("=== EXPERIMENT START - Initial Memory Check ===")
    initial_memory = check_memory()
    
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
        save_results(convert_results_to_csv([results]), r_filez[0])
        save_results([workers_selected], w_filez[0])
        
        # Log progress with memory info
        if epoch % 5 == 0:  # Every 5 epochs
            logger.info(f"=== PROGRESS UPDATE - Completed {epoch}/{args.get_num_epochs()} epochs ===")
            current_memory = check_memory()
            if initial_memory and current_memory:
                total_increase = current_memory['memory_mb'] - initial_memory['memory_mb']
                logger.info(f"Total memory increase since start: {total_increase:+.2f} MB")
                    

    # Final memory summary
    logger.info("=== EXPERIMENT END - Final Memory Check ===")
    final_memory = check_memory()
    if initial_memory and final_memory:
        total_increase = final_memory['memory_mb'] - initial_memory['memory_mb']
        logger.info(f"Total memory increase during experiment: {total_increase:+.2f} MB")

    # Print timing report and save timing statistics
    print_timing_report()
    
    # Generate experiment name for timing files
    experiment_name = f"dagtrust_experiment_{int(time.time())}"
    save_timing_statistics(experiment_name, args.get_num_epochs())
    
    logger.info("✅ Experiment completed with timing tracking")

    return convert_results_to_csv(epoch_test_set_results), worker_selection

def run_exp(replacement_method, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)
    global r_filez
    r_filez = results_files
    global w_filez
    w_filez = worker_selections_files
    # First, create a temporary args to get basic configuration
    temp_args = Arguments(logger)
    
    #TODO: mettere un numero <=X di workers per round
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 8  # Will be updated after DAG initialization
    }

    temp_args.set_round_worker_selection_strategy_kwargs(KWARGS)
    temp_args.set_client_selection_strategy(client_selection_strategy)
    temp_args.log()

    train_data_loader = load_train_data_loader(logger, temp_args)
    test_data_loader = load_test_data_loader(logger, temp_args)

    # Distribute batches equal volume IID - using default number of workers for now
    distributed_train_dataset = distribute_batches_equally(train_data_loader, temp_args.get_num_workers())
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    # Create initial clients to get their parameters
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, temp_args.get_batch_size())
    temp_clients = create_clients(temp_args, train_data_loaders, test_data_loader)
    
    # Extract initial parameters from all clients
    initial_parameters = {}
    for client in temp_clients:
        initial_parameters[client.get_client_index()] = client.get_nn_parameters()
    
    #GIACOMO: Initialize DAG with initial client parameters
    K, R, num_malicious, malicious_indices = init_DAG(initial_parameters, logger=logger)
    N = int(K/2)

    # Now update the configuration with actual values from DAG
    KWARGS["NUM_WORKERS_PER_ROUND"] = N

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_malicious)
    args.set_num_workers(K)
    args.set_num_epochs(R)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    # Redistribute data with correct number of workers
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    poisoned_workers = malicious_indices
    #poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    logger.info(f"Poisoned workers: {str(poisoned_workers)}")
    if torch.cuda.is_available():
        logger.info(f"CUDA is available, using GPU for training: {torch.cuda.get_device_name(0)}")
    else:
        logger.error("CUDA is not available, using CPU for training.")
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    clients = create_clients(args, train_data_loaders, test_data_loader)

    results, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    #save_results(results, results_files[0])
    #save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)


def save_timing_statistics(experiment_name, epoch):
    """
    Save timing statistics to CSV files for analysis.
    
    :param experiment_name: name of the experiment
    :type experiment_name: str
    :param epoch: final epoch number
    :type epoch: int
    """
    global ml_training_times, dag_operation_times, round_start_times, total_experiment_start_time
    
    import pandas as pd
    import os
    
    # Create timing data directory
    timing_dir = "timing_results"
    os.makedirs(timing_dir, exist_ok=True)
    
    # Calculate total experiment time
    total_experiment_time = time.time() - total_experiment_start_time if total_experiment_start_time else 0
    
    # Create timing summary
    timing_data = {
        'round': list(range(1, len(ml_training_times) + 1)),
        'ml_training_time': ml_training_times,
        'dag_operations_time': dag_operation_times,
        'total_round_time': [ml + dag for ml, dag in zip(ml_training_times, dag_operation_times)]
    }
    
    timing_df = pd.DataFrame(timing_data)
    
    # Save detailed timing data
    timing_file = os.path.join(timing_dir, f"{experiment_name}_timing_details.csv")
    timing_df.to_csv(timing_file, index=False)
    
    # Calculate and save summary statistics
    summary_stats = {
        'metric': [
            'total_experiment_time',
            'avg_ml_training_time_per_round',
            'avg_dag_operations_time_per_round', 
            'avg_total_round_time',
            'total_ml_training_time',
            'total_dag_operations_time',
            'ml_training_percentage',
            'dag_operations_percentage',
            'rounds_completed'
        ],
        'value': [
            total_experiment_time,
            np.mean(ml_training_times),
            np.mean(dag_operation_times),
            np.mean(timing_df['total_round_time']),
            np.sum(ml_training_times),
            np.sum(dag_operation_times),
            (np.sum(ml_training_times) / total_experiment_time * 100) if total_experiment_time > 0 else 0,
            (np.sum(dag_operation_times) / total_experiment_time * 100) if total_experiment_time > 0 else 0,
            len(ml_training_times)
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(timing_dir, f"{experiment_name}_timing_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"💾 Timing statistics saved:")
    logger.info(f"   Detailed: {timing_file}")
    logger.info(f"   Summary: {summary_file}")
    
    return timing_file, summary_file


def print_timing_report():
    """
    Print a comprehensive timing report to console.
    """
    global ml_training_times, dag_operation_times, total_experiment_start_time
    
    if not ml_training_times or not dag_operation_times:
        logger.warning("No timing data available for report")
        return
    
    total_experiment_time = time.time() - total_experiment_start_time if total_experiment_start_time else 0
    total_ml_time = np.sum(ml_training_times)
    total_dag_time = np.sum(dag_operation_times)
    total_federated_time = total_ml_time + total_dag_time
    
    logger.info("\n" + "="*80)
    logger.info("📊 TIMING PERFORMANCE REPORT")
    logger.info("="*80)
    logger.info(f"🕐 Total Experiment Time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)")
    logger.info(f"🔄 Rounds Completed: {len(ml_training_times)}")
    logger.info("")
    logger.info(f"🤖 ML Training:")
    logger.info(f"   Total Time: {total_ml_time:.2f} seconds ({total_ml_time/60:.2f} minutes)")
    logger.info(f"   Average per Round: {np.mean(ml_training_times):.3f} seconds")
    logger.info(f"   Min/Max per Round: {np.min(ml_training_times):.3f} / {np.max(ml_training_times):.3f} seconds")
    logger.info(f"   Percentage of Total: {(total_ml_time/total_experiment_time*100):.1f}%")
    logger.info("")
    logger.info(f"🔗 DAG Operations:")
    logger.info(f"   Total Time: {total_dag_time:.2f} seconds ({total_dag_time/60:.2f} minutes)")
    logger.info(f"   Average per Round: {np.mean(dag_operation_times):.3f} seconds")
    logger.info(f"   Min/Max per Round: {np.min(dag_operation_times):.3f} / {np.max(dag_operation_times):.3f} seconds")
    logger.info(f"   Percentage of Total: {(total_dag_time/total_experiment_time*100):.1f}%")
    logger.info("")
    logger.info(f"⚡ Performance Metrics:")
    logger.info(f"   Federated Learning Time: {total_federated_time:.2f} seconds ({(total_federated_time/total_experiment_time*100):.1f}% of total)")
    logger.info(f"   Overhead Time: {(total_experiment_time-total_federated_time):.2f} seconds ({((total_experiment_time-total_federated_time)/total_experiment_time*100):.1f}% of total)")
    logger.info(f"   ML/DAG Time Ratio: {(total_ml_time/total_dag_time):.2f}:1")
    logger.info("="*80)

    
