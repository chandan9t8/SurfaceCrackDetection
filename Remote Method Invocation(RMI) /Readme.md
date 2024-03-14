### Steps to implement the remote method invocation 

**Step 1**

connect to a remote server `server1` and compile bank and common classes

`javac ./bank/*.java`

`javac ./common/*.java`

**Step 2**

run the rmi registry

`rmiregistry`

**Step 3**

run the remote server on server1

`java bank.RemoteAccount`

**Step 4**

connect to server2 and compile `client` and `common` classes

`javac ./client/*.java`

`javac ./common/*.java`

**Step 5**

run the client on server2 

`java client.AccountClient`
