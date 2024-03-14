package common;

import java.rmi.Remote;

import java.rmi.RemoteException;

public interface Account extends Remote {
	
	Money getBalance() throws RemoteException;

    void deposit(Money amount) throws RemoteException;

    void withdraw(Money amount) throws RemoteException;

}