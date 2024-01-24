/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <grpcpp/grpcpp.h>

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "simple.grpc.pb.h"
#endif

ABSL_FLAG(std::string, target, "localhost:50051", "Server address");

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using simple::Requester;
using simple::HelloReply;
using simple::HelloRequest;
using simple::DataRequest;
using simple::DefaultResponse;

class RequesterClient {
public:
  RequesterClient(std::shared_ptr<Channel> channel)
      : stub_(Requester::NewStub(channel)) {}

  std::string SayRequest(const std::string &user,
                          int &user_id,
                          int &process_id,
                          int &node_id,
                          int &device_id,
                          int &function_id,
                          int &dims
                         ) {
    // Data we are sending to the server.
    DataRequest request;
    request.set_user_id(user_id);
    request.set_process_id(process_id);
    request.set_node_id(node_id);
    request.set_device_id(device_id);
    request.set_function_id(function_id);

    request.set_name(user);

    DefaultResponse reply;
    ClientContext context;

    // The actual RPC.
    Status status = stub_->PrintData(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }
  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string SayHello(const std::string &user) {
    // Data we are sending to the server.
    HelloRequest request;
    request.set_name(user);

    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->SayHello(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }

private:
  std::unique_ptr<Requester::Stub> stub_;
};

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint specified by
  // the argument "--target=" which is the only expected argument.
  std::string target_str = absl::GetFlag(FLAGS_target);
  // We indicate that the channel isn't authenticated (use of
  // InsecureChannelCredentials()).
  RequesterClient requester(
      grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));

  std::string user("world");

  int user_id, process_id, node_id, device_id, function_id, dims;

  std::cout << "Input variables: u,p,n,d,f,dim: " << std::flush;
  std::cin >> user_id >> process_id >> node_id >> device_id >> function_id >> dims;

  std::string reply = requester.SayRequest(user, user_id, process_id, node_id, device_id, function_id, dims);
  std::cout << "Requester received: " << reply << std::endl;

  return 0;
}
