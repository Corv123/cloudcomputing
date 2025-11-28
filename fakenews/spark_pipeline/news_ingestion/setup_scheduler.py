"""
Setup AWS EventBridge (CloudWatch Events) to run news ingestion pipeline hourly
"""

import boto3
import json
import argparse

def create_eventbridge_rule(
    rule_name: str = "news-ingestion-hourly",
    schedule: str = "rate(1 hour)",
    lambda_function_name: str = "news-ingestion-pipeline",
    region: str = "ap-southeast-2"
):
    """
    Create EventBridge rule to trigger news ingestion pipeline hourly
    
    Args:
        rule_name: Name of the EventBridge rule
        schedule: Schedule expression (e.g., "rate(1 hour)" or "cron(0 * * * ? *)")
        lambda_function_name: Name of Lambda function to trigger
        region: AWS region
    """
    events = boto3.client('events', region_name=region)
    lambda_client = boto3.client('lambda', region_name=region)
    
    print(f"Creating EventBridge rule: {rule_name}")
    print(f"Schedule: {schedule}")
    print()
    
    # Create rule
    try:
        response = events.put_rule(
            Name=rule_name,
            ScheduleExpression=schedule,
            State='ENABLED',
            Description='Trigger news ingestion pipeline hourly'
        )
        print(f"[OK] Created EventBridge rule: {rule_name}")
        print(f"     Rule ARN: {response['RuleArn']}")
    except Exception as e:
        print(f"[ERROR] Failed to create rule: {e}")
        return False
    
    # Add Lambda permission
    try:
        lambda_client.add_permission(
            FunctionName=lambda_function_name,
            StatementId=f'{rule_name}-event',
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=response['RuleArn']
        )
        print(f"[OK] Added Lambda permission")
    except Exception as e:
        if 'already exists' in str(e).lower():
            print(f"[OK] Lambda permission already exists")
        else:
            print(f"[WARN] Could not add Lambda permission: {e}")
    
    # Add Lambda as target
    try:
        events.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': f'arn:aws:lambda:{region}:979207815314:function:{lambda_function_name}'
                }
            ]
        )
        print(f"[OK] Added Lambda as target")
    except Exception as e:
        print(f"[ERROR] Failed to add target: {e}")
        return False
    
    print()
    print("=" * 80)
    print("SCHEDULER SETUP COMPLETE")
    print("=" * 80)
    print(f"Rule: {rule_name}")
    print(f"Schedule: {schedule}")
    print(f"Target: {lambda_function_name}")
    print()
    print("The pipeline will now run automatically every hour.")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup EventBridge scheduler for news ingestion")
    parser.add_argument("--rule_name", default="news-ingestion-hourly",
                       help="EventBridge rule name")
    parser.add_argument("--schedule", default="rate(1 hour)",
                       help="Schedule expression (e.g., 'rate(1 hour)' or 'cron(0 * * * ? *)')")
    parser.add_argument("--lambda_function", default="news-ingestion-pipeline",
                       help="Lambda function name")
    parser.add_argument("--region", default="ap-southeast-2",
                       help="AWS region")
    
    args = parser.parse_args()
    
    create_eventbridge_rule(
        rule_name=args.rule_name,
        schedule=args.schedule,
        lambda_function_name=args.lambda_function,
        region=args.region
    )

