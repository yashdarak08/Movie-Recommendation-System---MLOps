variable "aws_region" {
  description = "AWS region to deploy resources"
  default     = "us-west-2"
}

variable "aws_ami" {
  description = "AMI ID for the EC2 instance"
  default     = "ami-0abcdef1234567890"  # Replace with a valid AMI ID
}

variable "aws_instance_type" {
  description = "EC2 instance type"
  default     = "t2.large"
}
