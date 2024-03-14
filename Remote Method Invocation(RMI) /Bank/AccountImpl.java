import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

import common.Account;

public class AccountImpl extends UnicastRemoteObject implements Account {
	
	private String name;
	private Money balance;
	
	AccountImpl() throws RemoteException {
		super();
	}
	
	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public void setBalance(Money balance) {
		this.balance = balance;
	}


	@Override
	public Money getBalance() throws RemoteException {
		return balance;
	}

	@Override
	public void deposit(Money amount) throws RemoteException {
		balance.add(amount);
		
	}

	@Override
	public void withdraw(Money amount) throws RemoteException {
		balance.subtract(amount);
		
	}

}